#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <zlib.h>

#define BGZF_MAX_UNCOMPRESSED 65280
#define BCF_FLOAT_MISSING 0x7F800001u

typedef struct {
    unsigned char *data;
    Py_ssize_t len;
    Py_ssize_t cap;
} ByteBuffer;

typedef struct {
    FILE *fp;
    unsigned char block[BGZF_MAX_UNCOMPRESSED];
    size_t used;
    int compression_level;
} BGZFWriter;

static void
bb_free(ByteBuffer *buf)
{
    PyMem_Free(buf->data);
    buf->data = NULL;
    buf->len = 0;
    buf->cap = 0;
}

static int
bb_reserve(ByteBuffer *buf, Py_ssize_t extra)
{
    Py_ssize_t needed;
    Py_ssize_t new_cap;
    unsigned char *new_data;

    if (extra < 0 || buf->len > PY_SSIZE_T_MAX - extra) {
        PyErr_SetString(PyExc_MemoryError, "BCF record buffer is too large.");
        return -1;
    }
    needed = buf->len + extra;
    if (needed <= buf->cap) {
        return 0;
    }
    new_cap = buf->cap > 0 ? buf->cap : 4096;
    while (new_cap < needed) {
        if (new_cap > PY_SSIZE_T_MAX / 2) {
            PyErr_SetString(PyExc_MemoryError, "BCF record buffer is too large.");
            return -1;
        }
        new_cap *= 2;
    }
    new_data = PyMem_Realloc(buf->data, (size_t)new_cap);
    if (new_data == NULL) {
        PyErr_NoMemory();
        return -1;
    }
    buf->data = new_data;
    buf->cap = new_cap;
    return 0;
}

static int
bb_append(ByteBuffer *buf, const void *src, Py_ssize_t len)
{
    if (len < 0) {
        PyErr_SetString(PyExc_ValueError, "Negative byte count while writing BCF.");
        return -1;
    }
    if (bb_reserve(buf, len) < 0) {
        return -1;
    }
    memcpy(buf->data + buf->len, src, (size_t)len);
    buf->len += len;
    return 0;
}

static int
bb_append_u8(ByteBuffer *buf, uint8_t value)
{
    if (bb_reserve(buf, 1) < 0) {
        return -1;
    }
    buf->data[buf->len++] = value;
    return 0;
}

static int
bb_append_u16_le(ByteBuffer *buf, uint16_t value)
{
    unsigned char bytes[2];
    bytes[0] = (unsigned char)(value & 0xffu);
    bytes[1] = (unsigned char)((value >> 8) & 0xffu);
    return bb_append(buf, bytes, 2);
}

static int
bb_append_u32_le(ByteBuffer *buf, uint32_t value)
{
    unsigned char bytes[4];
    bytes[0] = (unsigned char)(value & 0xffu);
    bytes[1] = (unsigned char)((value >> 8) & 0xffu);
    bytes[2] = (unsigned char)((value >> 16) & 0xffu);
    bytes[3] = (unsigned char)((value >> 24) & 0xffu);
    return bb_append(buf, bytes, 4);
}

static int
bb_append_i32_le(ByteBuffer *buf, int32_t value)
{
    return bb_append_u32_le(buf, (uint32_t)value);
}

static int
bb_append_typed_descriptor(ByteBuffer *buf, Py_ssize_t n_values, int type_code)
{
    if (n_values < 0) {
        PyErr_SetString(PyExc_ValueError, "BCF typed value length cannot be negative.");
        return -1;
    }
    if (n_values < 15) {
        return bb_append_u8(buf, (uint8_t)((n_values << 4) | type_code));
    }
    if (bb_append_u8(buf, (uint8_t)((15 << 4) | type_code)) < 0) {
        return -1;
    }
    if (n_values <= 127) {
        return bb_append(buf, (unsigned char[]){0x11u, (unsigned char)n_values}, 2);
    }
    if (n_values <= 32767) {
        if (bb_append_u8(buf, 0x12u) < 0) {
            return -1;
        }
        return bb_append_u16_le(buf, (uint16_t)n_values);
    }
    if (n_values <= INT32_MAX) {
        if (bb_append_u8(buf, 0x13u) < 0) {
            return -1;
        }
        return bb_append_u32_le(buf, (uint32_t)n_values);
    }
    PyErr_SetString(PyExc_ValueError, "BCF typed value length exceeds int32 range.");
    return -1;
}

static int
bb_append_typed_string(ByteBuffer *buf, const char *value, Py_ssize_t len)
{
    if (bb_append_typed_descriptor(buf, len, 7) < 0) {
        return -1;
    }
    if (len == 0) {
        return 0;
    }
    return bb_append(buf, value, len);
}

static int
bb_append_typed_int_scalar(ByteBuffer *buf, int32_t value)
{
    if (value >= -120 && value <= 127) {
        unsigned char raw = (unsigned char)(value & 0xff);
        return bb_append(buf, (unsigned char[]){0x11u, raw}, 2);
    }
    if (value >= -32760 && value <= 32767) {
        if (bb_append_u8(buf, 0x12u) < 0) {
            return -1;
        }
        return bb_append_u16_le(buf, (uint16_t)value);
    }
    if (bb_append_u8(buf, 0x13u) < 0) {
        return -1;
    }
    return bb_append_u32_le(buf, (uint32_t)value);
}

static int
bgzf_open(BGZFWriter *writer, const char *filename, int compression_level)
{
    writer->fp = fopen(filename, "wb");
    writer->used = 0;
    writer->compression_level = compression_level;
    if (writer->fp == NULL) {
        PyErr_SetFromErrnoWithFilename(PyExc_OSError, filename);
        return -1;
    }
    return 0;
}

static int
bgzf_flush(BGZFWriter *writer)
{
    z_stream stream;
    unsigned long bound;
    unsigned char *compressed = NULL;
    unsigned long compressed_len;
    uint32_t crc;
    uint32_t isize;
    uint16_t bsize;
    unsigned char header[18] = {
        0x1f, 0x8b, 0x08, 0x04, 0x00, 0x00, 0x00, 0x00, 0x00, 0xff,
        0x06, 0x00, 'B', 'C', 0x02, 0x00, 0x00, 0x00
    };
    unsigned char trailer[8];
    int zret;
    size_t total_size;

    if (writer->used == 0) {
        return 0;
    }

    bound = compressBound((uLong)writer->used);
    compressed = PyMem_Malloc(bound);
    if (compressed == NULL) {
        PyErr_NoMemory();
        return -1;
    }

    memset(&stream, 0, sizeof(stream));
    zret = deflateInit2(
        &stream,
        writer->compression_level,
        Z_DEFLATED,
        -15,
        8,
        Z_DEFAULT_STRATEGY
    );
    if (zret != Z_OK) {
        PyMem_Free(compressed);
        PyErr_SetString(PyExc_RuntimeError, "zlib deflateInit2 failed while writing BGZF.");
        return -1;
    }

    stream.next_in = writer->block;
    stream.avail_in = (uInt)writer->used;
    stream.next_out = compressed;
    stream.avail_out = (uInt)bound;
    zret = deflate(&stream, Z_FINISH);
    if (zret != Z_STREAM_END) {
        deflateEnd(&stream);
        PyMem_Free(compressed);
        PyErr_SetString(PyExc_RuntimeError, "zlib deflate failed while writing BGZF.");
        return -1;
    }
    compressed_len = stream.total_out;
    deflateEnd(&stream);

    total_size = 18u + (size_t)compressed_len + 8u;
    if (total_size > 65536u) {
        PyMem_Free(compressed);
        PyErr_SetString(PyExc_RuntimeError, "Compressed BGZF block exceeds 64 KiB.");
        return -1;
    }
    bsize = (uint16_t)(total_size - 1u);
    header[16] = (unsigned char)(bsize & 0xffu);
    header[17] = (unsigned char)((bsize >> 8) & 0xffu);

    crc = crc32(0L, Z_NULL, 0);
    crc = crc32(crc, writer->block, (uInt)writer->used);
    isize = (uint32_t)writer->used;
    trailer[0] = (unsigned char)(crc & 0xffu);
    trailer[1] = (unsigned char)((crc >> 8) & 0xffu);
    trailer[2] = (unsigned char)((crc >> 16) & 0xffu);
    trailer[3] = (unsigned char)((crc >> 24) & 0xffu);
    trailer[4] = (unsigned char)(isize & 0xffu);
    trailer[5] = (unsigned char)((isize >> 8) & 0xffu);
    trailer[6] = (unsigned char)((isize >> 16) & 0xffu);
    trailer[7] = (unsigned char)((isize >> 24) & 0xffu);

    if (fwrite(header, 1, sizeof(header), writer->fp) != sizeof(header)
        || fwrite(compressed, 1, (size_t)compressed_len, writer->fp) != (size_t)compressed_len
        || fwrite(trailer, 1, sizeof(trailer), writer->fp) != sizeof(trailer)) {
        PyMem_Free(compressed);
        PyErr_SetFromErrno(PyExc_OSError);
        return -1;
    }

    PyMem_Free(compressed);
    writer->used = 0;
    return 0;
}

static int
bgzf_write(BGZFWriter *writer, const unsigned char *data, Py_ssize_t len)
{
    Py_ssize_t offset = 0;
    while (offset < len) {
        size_t space = BGZF_MAX_UNCOMPRESSED - writer->used;
        size_t take = (size_t)(len - offset);
        if (take > space) {
            take = space;
        }
        memcpy(writer->block + writer->used, data + offset, take);
        writer->used += take;
        offset += (Py_ssize_t)take;
        if (writer->used == BGZF_MAX_UNCOMPRESSED) {
            if (bgzf_flush(writer) < 0) {
                return -1;
            }
        }
    }
    return 0;
}

static int
bgzf_close(BGZFWriter *writer)
{
    static const unsigned char eof_block[28] = {
        0x1f, 0x8b, 0x08, 0x04, 0x00, 0x00, 0x00, 0x00,
        0x00, 0xff, 0x06, 0x00, 'B', 'C', 0x02, 0x00,
        0x1b, 0x00, 0x03, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00
    };
    int close_result;

    if (writer->fp == NULL) {
        return 0;
    }
    if (bgzf_flush(writer) < 0) {
        fclose(writer->fp);
        writer->fp = NULL;
        return -1;
    }
    if (fwrite(eof_block, 1, sizeof(eof_block), writer->fp) != sizeof(eof_block)) {
        fclose(writer->fp);
        writer->fp = NULL;
        PyErr_SetFromErrno(PyExc_OSError);
        return -1;
    }
    close_result = fclose(writer->fp);
    writer->fp = NULL;
    if (close_result != 0) {
        PyErr_SetFromErrno(PyExc_OSError);
        return -1;
    }
    return 0;
}

static void
bgzf_abort(BGZFWriter *writer)
{
    if (writer->fp != NULL) {
        fclose(writer->fp);
        writer->fp = NULL;
    }
}

static int
as_sequence_fast(PyObject *obj, const char *name, PyObject **fast, Py_ssize_t expected_len)
{
    *fast = PySequence_Fast(obj, name);
    if (*fast == NULL) {
        return -1;
    }
    if (PySequence_Fast_GET_SIZE(*fast) != expected_len) {
        Py_DECREF(*fast);
        *fast = NULL;
        PyErr_Format(PyExc_ValueError, "%s length must match number of variants.", name);
        return -1;
    }
    return 0;
}

static int
unicode_item(PyObject *fast, Py_ssize_t idx, const char *field_name, const char **data, Py_ssize_t *len)
{
    PyObject *item = PySequence_Fast_GET_ITEM(fast, idx);
    if (!PyUnicode_Check(item)) {
        PyErr_Format(PyExc_TypeError, "%s entries must be strings.", field_name);
        return -1;
    }
    *data = PyUnicode_AsUTF8AndSize(item, len);
    if (*data == NULL) {
        return -1;
    }
    return 0;
}

static int
bytes_item(PyObject *fast, Py_ssize_t idx, const char *field_name, const char **data, Py_ssize_t *len)
{
    PyObject *item = PySequence_Fast_GET_ITEM(fast, idx);
    if (!PyBytes_Check(item)) {
        PyErr_Format(PyExc_TypeError, "%s entries must be bytes.", field_name);
        return -1;
    }
    if (PyBytes_AsStringAndSize(item, (char **)data, len) < 0) {
        return -1;
    }
    return 0;
}

static int
alt_allele_count(const char *alt, Py_ssize_t len, uint32_t *count)
{
    Py_ssize_t i;
    uint32_t n = 1;

    if (len == 0 || (len == 1 && alt[0] == '.')) {
        *count = 1;
        return 0;
    }
    for (i = 0; i < len; i++) {
        if (alt[i] == ',') {
            n++;
        }
    }
    *count = 1u + n;
    return 0;
}

static int
append_alt_alleles(ByteBuffer *buf, const char *alt, Py_ssize_t len)
{
    Py_ssize_t start = 0;
    Py_ssize_t i;

    if (len == 0 || (len == 1 && alt[0] == '.')) {
        return 0;
    }
    for (i = 0; i <= len; i++) {
        if (i == len || alt[i] == ',') {
            if (i == start) {
                PyErr_SetString(PyExc_ValueError, "ALT contains an empty allele.");
                return -1;
            }
            if (bb_append_typed_string(buf, alt + start, i - start) < 0) {
                return -1;
            }
            start = i + 1;
        }
    }
    return 0;
}

static int
gt_type_code_from_size(int gt_type_size)
{
    if (gt_type_size == 1) {
        return 1;
    }
    if (gt_type_size == 2) {
        return 2;
    }
    if (gt_type_size == 4) {
        return 3;
    }
    return -1;
}

static int
append_gt_value(ByteBuffer *buf, int allele, int allele_index, int gt_type_size, int phased)
{
    uint32_t raw;
    uint32_t max_value;

    if (allele < 0) {
        raw = 0;
    } else {
        raw = (((uint32_t)allele + 1u) << 1) | ((phased && allele_index > 0) ? 1u : 0u);
    }

    if (gt_type_size == 1) {
        max_value = 0x7fu;
    } else if (gt_type_size == 2) {
        max_value = 0x7fffu;
    } else {
        max_value = 0x7fffffffu;
    }
    if (raw > max_value) {
        PyErr_SetString(PyExc_ValueError, "Genotype allele index is too large for BCF GT encoding.");
        return -1;
    }

    if (gt_type_size == 1) {
        return bb_append_u8(buf, (uint8_t)raw);
    }
    if (gt_type_size == 2) {
        return bb_append_u16_le(buf, (uint16_t)raw);
    }
    return bb_append_u32_le(buf, raw);
}

static int
append_gt_data(
    ByteBuffer *buf,
    const int16_t *gt,
    Py_ssize_t variant_idx,
    Py_ssize_t n_samples,
    int gt_mode,
    int gt_width,
    int gt_type_size,
    int phased
)
{
    Py_ssize_t sample_idx;
    int allele;

    if (gt_mode == 2) {
        const int16_t *row = gt + variant_idx * n_samples;
        for (sample_idx = 0; sample_idx < n_samples; sample_idx++) {
            int dosage = (int)row[sample_idx];
            int a0;
            int a1;
            if (dosage < 0) {
                a0 = -1;
                a1 = -1;
            } else if (dosage == 0) {
                a0 = 0;
                a1 = 0;
            } else if (dosage == 1) {
                a0 = 0;
                a1 = 1;
            } else if (dosage == 2) {
                a0 = 1;
                a1 = 1;
            } else {
                PyErr_SetString(PyExc_ValueError, "2D BCF genotypes must contain hard-call dosages 0, 1, 2, or missing values.");
                return -1;
            }
            if (append_gt_value(buf, a0, 0, gt_type_size, phased) < 0
                || append_gt_value(buf, a1, 1, gt_type_size, phased) < 0) {
                return -1;
            }
        }
        return 0;
    }

    for (sample_idx = 0; sample_idx < n_samples; sample_idx++) {
        int allele_idx;
        const int16_t *sample = gt + (variant_idx * n_samples + sample_idx) * gt_width;
        for (allele_idx = 0; allele_idx < gt_width; allele_idx++) {
            allele = (int)sample[allele_idx];
            if (append_gt_value(buf, allele, allele_idx, gt_type_size, phased) < 0) {
                return -1;
            }
        }
    }
    return 0;
}

static int
append_gp_data(
    ByteBuffer *buf,
    const float *gp,
    Py_ssize_t variant_idx,
    Py_ssize_t n_samples,
    Py_ssize_t n_probs
)
{
    Py_ssize_t sample_idx;
    Py_ssize_t prob_idx;
    const float *row = gp + variant_idx * n_samples * n_probs;

    for (sample_idx = 0; sample_idx < n_samples; sample_idx++) {
        for (prob_idx = 0; prob_idx < n_probs; prob_idx++) {
            float value = row[sample_idx * n_probs + prob_idx];
            if (isnan(value)) {
                if (bb_append_u32_le(buf, BCF_FLOAT_MISSING) < 0) {
                    return -1;
                }
            } else {
                union {
                    float f;
                    uint32_t u;
                } bits;
                bits.f = value;
                if (bb_append_u32_le(buf, bits.u) < 0) {
                    return -1;
                }
            }
        }
    }
    return 0;
}

static int
write_u32_to_bgzf(BGZFWriter *writer, uint32_t value)
{
    unsigned char bytes[4];
    bytes[0] = (unsigned char)(value & 0xffu);
    bytes[1] = (unsigned char)((value >> 8) & 0xffu);
    bytes[2] = (unsigned char)((value >> 16) & 0xffu);
    bytes[3] = (unsigned char)((value >> 24) & 0xffu);
    return bgzf_write(writer, bytes, 4);
}

static PyObject *
write_bcf(PyObject *self, PyObject *args, PyObject *kwargs)
{
    PyObject *filename_obj;
    PyObject *header_obj;
    PyObject *chrom_obj;
    PyObject *pos_obj;
    PyObject *ids_obj;
    PyObject *refs_obj;
    PyObject *alts_obj;
    PyObject *qual_obj;
    PyObject *filter_obj;
    PyObject *info_obj;
    PyObject *n_info_obj;
    PyObject *gt_obj;
    PyObject *gp_obj;
    int gt_mode;
    int gt_width;
    int gt_type_size;
    int phased;
    int gt_idx;
    int gp_idx;
    int compression_level;
    static char *kwlist[] = {
        "filename", "header_text", "chrom_ids", "positions", "ids", "refs", "alts",
        "qual_bits", "filter_blobs", "info_blobs", "n_info", "genotypes",
        "gt_mode", "gt_width", "gt_type_size", "gp", "phased", "gt_idx",
        "gp_idx", "compression_level", NULL
    };

    const char *filename;
    const char *header_text;
    Py_ssize_t header_len;
    uint32_t header_bcf_len;
    unsigned char prefix[9] = {'B', 'C', 'F', 0x02, 0x02, 0, 0, 0, 0};
    unsigned char zero = 0;
    Py_buffer chrom_view = {0};
    Py_buffer pos_view = {0};
    Py_buffer qual_view = {0};
    Py_buffer n_info_view = {0};
    Py_buffer gt_view = {0};
    Py_buffer gp_view = {0};
    int have_gt_view = 0;
    int have_gp_view = 0;
    PyObject *ids_fast = NULL;
    PyObject *refs_fast = NULL;
    PyObject *alts_fast = NULL;
    PyObject *filter_fast = NULL;
    PyObject *info_fast = NULL;
    Py_ssize_t n_variants;
    Py_ssize_t n_samples = 0;
    Py_ssize_t gp_n_samples = 0;
    Py_ssize_t n_probs = 0;
    const int32_t *chrom_ids;
    const int64_t *positions;
    const uint32_t *qual_bits;
    const uint16_t *n_info_values;
    const int16_t *gt_values = NULL;
    const float *gp_values = NULL;
    BGZFWriter writer = {0};
    ByteBuffer shared = {0};
    ByteBuffer indiv = {0};
    Py_ssize_t i;
    int result = -1;

    (void)self;

    if (!PyArg_ParseTupleAndKeywords(
            args,
            kwargs,
            "OOOOOOOOOOOOiiiOpiii",
            kwlist,
            &filename_obj,
            &header_obj,
            &chrom_obj,
            &pos_obj,
            &ids_obj,
            &refs_obj,
            &alts_obj,
            &qual_obj,
            &filter_obj,
            &info_obj,
            &n_info_obj,
            &gt_obj,
            &gt_mode,
            &gt_width,
            &gt_type_size,
            &gp_obj,
            &phased,
            &gt_idx,
            &gp_idx,
            &compression_level)) {
        return NULL;
    }

    filename = PyUnicode_AsUTF8(filename_obj);
    if (filename == NULL) {
        return NULL;
    }
    header_text = PyUnicode_AsUTF8AndSize(header_obj, &header_len);
    if (header_text == NULL) {
        return NULL;
    }
    if (header_len < 0 || header_len >= UINT32_MAX) {
        PyErr_SetString(PyExc_ValueError, "BCF header is too large.");
        return NULL;
    }
    header_bcf_len = (uint32_t)header_len + 1u;

    if (PyObject_GetBuffer(chrom_obj, &chrom_view, PyBUF_ND | PyBUF_C_CONTIGUOUS) < 0
        || PyObject_GetBuffer(pos_obj, &pos_view, PyBUF_ND | PyBUF_C_CONTIGUOUS) < 0
        || PyObject_GetBuffer(qual_obj, &qual_view, PyBUF_ND | PyBUF_C_CONTIGUOUS) < 0
        || PyObject_GetBuffer(n_info_obj, &n_info_view, PyBUF_ND | PyBUF_C_CONTIGUOUS) < 0) {
        goto cleanup;
    }

    if (chrom_view.ndim != 1 || pos_view.ndim != 1 || qual_view.ndim != 1 || n_info_view.ndim != 1
        || chrom_view.itemsize != 4 || pos_view.itemsize != 8
        || qual_view.itemsize != 4 || n_info_view.itemsize != 2) {
        PyErr_SetString(PyExc_ValueError, "BCF writer received variant arrays with unexpected shape or dtype.");
        goto cleanup;
    }

    n_variants = chrom_view.shape[0];
    if (pos_view.shape[0] != n_variants || qual_view.shape[0] != n_variants || n_info_view.shape[0] != n_variants) {
        PyErr_SetString(PyExc_ValueError, "BCF writer variant arrays must have matching lengths.");
        goto cleanup;
    }

    if (as_sequence_fast(ids_obj, "ids", &ids_fast, n_variants) < 0
        || as_sequence_fast(refs_obj, "refs", &refs_fast, n_variants) < 0
        || as_sequence_fast(alts_obj, "alts", &alts_fast, n_variants) < 0
        || as_sequence_fast(filter_obj, "filter_blobs", &filter_fast, n_variants) < 0
        || as_sequence_fast(info_obj, "info_blobs", &info_fast, n_variants) < 0) {
        goto cleanup;
    }

    if (gt_mode != 0) {
        int expected_ndim = gt_mode == 2 ? 2 : 3;
        if (PyObject_GetBuffer(gt_obj, &gt_view, PyBUF_ND | PyBUF_C_CONTIGUOUS) < 0) {
            goto cleanup;
        }
        have_gt_view = 1;
        if (gt_view.ndim != expected_ndim || gt_view.itemsize != 2 || gt_view.shape[0] != n_variants) {
            PyErr_SetString(PyExc_ValueError, "BCF writer received genotypes with unexpected shape or dtype.");
            goto cleanup;
        }
        n_samples = gt_view.shape[1];
        if (gt_mode == 3 && gt_view.shape[2] != gt_width) {
            PyErr_SetString(PyExc_ValueError, "BCF genotype allele width does not match input array.");
            goto cleanup;
        }
        if (gt_mode == 2 && gt_width != 2) {
            PyErr_SetString(PyExc_ValueError, "2D BCF genotype dosages must be encoded as diploid GT.");
            goto cleanup;
        }
        if (gt_width != 1 && gt_width != 2) {
            PyErr_SetString(PyExc_ValueError, "BCFWriter supports haploid or diploid GT fields.");
            goto cleanup;
        }
        if (gt_type_code_from_size(gt_type_size) < 0) {
            PyErr_SetString(PyExc_ValueError, "Unsupported BCF GT integer width.");
            goto cleanup;
        }
        gt_values = (const int16_t *)gt_view.buf;
    }

    if (gp_obj != Py_None) {
        if (PyObject_GetBuffer(gp_obj, &gp_view, PyBUF_ND | PyBUF_C_CONTIGUOUS) < 0) {
            goto cleanup;
        }
        have_gp_view = 1;
        if (gp_view.ndim != 3 || gp_view.itemsize != 4 || gp_view.shape[0] != n_variants) {
            PyErr_SetString(PyExc_ValueError, "BCF writer received GP with unexpected shape or dtype.");
            goto cleanup;
        }
        gp_n_samples = gp_view.shape[1];
        n_probs = gp_view.shape[2];
        if (n_probs < 1) {
            PyErr_SetString(PyExc_ValueError, "BCF FORMAT/GP must contain at least one probability per sample.");
            goto cleanup;
        }
        if (n_samples == 0) {
            n_samples = gp_n_samples;
        } else if (gp_n_samples != n_samples) {
            PyErr_SetString(PyExc_ValueError, "GT and GP sample counts do not match.");
            goto cleanup;
        }
        gp_values = (const float *)gp_view.buf;
    }

    chrom_ids = (const int32_t *)chrom_view.buf;
    positions = (const int64_t *)pos_view.buf;
    qual_bits = (const uint32_t *)qual_view.buf;
    n_info_values = (const uint16_t *)n_info_view.buf;

    if (compression_level < 0 || compression_level > 9) {
        PyErr_SetString(PyExc_ValueError, "BGZF compression_level must be between 0 and 9.");
        goto cleanup;
    }

    if (bgzf_open(&writer, filename, compression_level) < 0) {
        goto cleanup;
    }

    prefix[5] = (unsigned char)(header_bcf_len & 0xffu);
    prefix[6] = (unsigned char)((header_bcf_len >> 8) & 0xffu);
    prefix[7] = (unsigned char)((header_bcf_len >> 16) & 0xffu);
    prefix[8] = (unsigned char)((header_bcf_len >> 24) & 0xffu);
    if (bgzf_write(&writer, prefix, 9) < 0
        || bgzf_write(&writer, (const unsigned char *)header_text, header_len) < 0
        || bgzf_write(&writer, &zero, 1) < 0) {
        goto cleanup;
    }

    for (i = 0; i < n_variants; i++) {
        const char *id;
        const char *ref;
        const char *alt;
        const char *filter_blob;
        const char *info_blob;
        Py_ssize_t id_len;
        Py_ssize_t ref_len;
        Py_ssize_t alt_len;
        Py_ssize_t filter_len;
        Py_ssize_t info_len;
        uint32_t n_alleles;
        uint32_t n_info = (uint32_t)n_info_values[i];
        uint32_t n_fmt = 0;
        int32_t pos0;
        uint32_t n_alleles_info;
        uint32_t n_fmt_samples;

        shared.len = 0;
        indiv.len = 0;

        if (unicode_item(ids_fast, i, "ids", &id, &id_len) < 0
            || unicode_item(refs_fast, i, "refs", &ref, &ref_len) < 0
            || unicode_item(alts_fast, i, "alts", &alt, &alt_len) < 0
            || bytes_item(filter_fast, i, "filter_blobs", &filter_blob, &filter_len) < 0
            || bytes_item(info_fast, i, "info_blobs", &info_blob, &info_len) < 0) {
            goto cleanup;
        }

        if (positions[i] < 1 || positions[i] > INT32_MAX) {
            PyErr_SetString(PyExc_ValueError, "BCF variant positions must be in the 1-based int32 range.");
            goto cleanup;
        }
        if (ref_len <= 0) {
            PyErr_SetString(PyExc_ValueError, "REF alleles must be non-empty.");
            goto cleanup;
        }
        if (alt_allele_count(alt, alt_len, &n_alleles) < 0) {
            goto cleanup;
        }
        if (n_alleles > 0xffffu || n_info > 0xffffu || n_samples > 0xffffff) {
            PyErr_SetString(PyExc_ValueError, "BCF record exceeds representable allele, INFO, or sample count.");
            goto cleanup;
        }

        if (n_samples > 0 && gt_mode != 0) {
            n_fmt++;
        }
        if (n_samples > 0 && have_gp_view) {
            n_fmt++;
        }

        pos0 = (int32_t)(positions[i] - 1);
        n_alleles_info = (n_alleles << 16) | n_info;
        n_fmt_samples = (n_fmt << 24) | (uint32_t)n_samples;

        if (bb_append_i32_le(&shared, chrom_ids[i]) < 0
            || bb_append_i32_le(&shared, pos0) < 0
            || bb_append_i32_le(&shared, (int32_t)ref_len) < 0
            || bb_append_u32_le(&shared, qual_bits[i]) < 0
            || bb_append_u32_le(&shared, n_alleles_info) < 0
            || bb_append_u32_le(&shared, n_fmt_samples) < 0) {
            goto cleanup;
        }

        if (id_len == 1 && id[0] == '.') {
            id_len = 0;
        }
        if (bb_append_typed_string(&shared, id, id_len) < 0
            || bb_append_typed_string(&shared, ref, ref_len) < 0
            || append_alt_alleles(&shared, alt, alt_len) < 0
            || bb_append(&shared, filter_blob, filter_len) < 0
            || bb_append(&shared, info_blob, info_len) < 0) {
            goto cleanup;
        }

        if (n_samples > 0 && gt_mode != 0) {
            int gt_type_code = gt_type_code_from_size(gt_type_size);
            if (bb_append_typed_int_scalar(&indiv, gt_idx) < 0
                || bb_append_typed_descriptor(&indiv, gt_width, gt_type_code) < 0
                || append_gt_data(&indiv, gt_values, i, n_samples, gt_mode, gt_width, gt_type_size, phased) < 0) {
                goto cleanup;
            }
        }
        if (n_samples > 0 && have_gp_view) {
            if (bb_append_typed_int_scalar(&indiv, gp_idx) < 0
                || bb_append_typed_descriptor(&indiv, n_probs, 5) < 0
                || append_gp_data(&indiv, gp_values, i, n_samples, n_probs) < 0) {
                goto cleanup;
            }
        }

        if (shared.len > UINT32_MAX || indiv.len > UINT32_MAX) {
            PyErr_SetString(PyExc_ValueError, "BCF record section exceeds uint32 length.");
            goto cleanup;
        }
        if (write_u32_to_bgzf(&writer, (uint32_t)shared.len) < 0
            || write_u32_to_bgzf(&writer, (uint32_t)indiv.len) < 0
            || bgzf_write(&writer, shared.data, shared.len) < 0
            || bgzf_write(&writer, indiv.data, indiv.len) < 0) {
            goto cleanup;
        }
    }

    if (bgzf_close(&writer) < 0) {
        goto cleanup;
    }
    result = 0;

cleanup:
    if (result != 0) {
        bgzf_abort(&writer);
    }
    bb_free(&shared);
    bb_free(&indiv);
    Py_XDECREF(ids_fast);
    Py_XDECREF(refs_fast);
    Py_XDECREF(alts_fast);
    Py_XDECREF(filter_fast);
    Py_XDECREF(info_fast);
    if (chrom_view.buf != NULL) {
        PyBuffer_Release(&chrom_view);
    }
    if (pos_view.buf != NULL) {
        PyBuffer_Release(&pos_view);
    }
    if (qual_view.buf != NULL) {
        PyBuffer_Release(&qual_view);
    }
    if (n_info_view.buf != NULL) {
        PyBuffer_Release(&n_info_view);
    }
    if (have_gt_view) {
        PyBuffer_Release(&gt_view);
    }
    if (have_gp_view) {
        PyBuffer_Release(&gp_view);
    }
    if (result != 0) {
        return NULL;
    }
    Py_RETURN_NONE;
}

static PyMethodDef BCFWriterMethods[] = {
    {"write_bcf", (PyCFunction)write_bcf, METH_VARARGS | METH_KEYWORDS, "Write a BCF2.2 file with BGZF compression."},
    {NULL, NULL, 0, NULL}
};

static struct PyModuleDef moduledef = {
    PyModuleDef_HEAD_INIT,
    "_bcf",
    "Native BCF writer helpers.",
    -1,
    BCFWriterMethods
};

PyMODINIT_FUNC
PyInit__bcf(void)
{
    return PyModule_Create(&moduledef);
}
