#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <limits.h>
#include <stdint.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <zlib.h>

#if defined(__x86_64__) && defined(__GNUC__)
#include <immintrin.h>
#define BGEN_HAVE_AVX2 1
#else
#define BGEN_HAVE_AVX2 0
#endif

static uint16_t
read_u16_le(const unsigned char *data)
{
    return (uint16_t)data[0] | ((uint16_t)data[1] << 8);
}

static uint32_t
read_u32_le(const unsigned char *data)
{
    return (uint32_t)data[0]
        | ((uint32_t)data[1] << 8)
        | ((uint32_t)data[2] << 16)
        | ((uint32_t)data[3] << 24);
}

static void
write_u16_le(unsigned char *data, uint16_t value)
{
    data[0] = (unsigned char)(value & 0xFF);
    data[1] = (unsigned char)((value >> 8) & 0xFF);
}

static void
write_u32_le(unsigned char *data, uint32_t value)
{
    data[0] = (unsigned char)(value & 0xFF);
    data[1] = (unsigned char)((value >> 8) & 0xFF);
    data[2] = (unsigned char)((value >> 16) & 0xFF);
    data[3] = (unsigned char)((value >> 24) & 0xFF);
}

static uint64_t
choose_u64(unsigned int n, unsigned int k)
{
    uint64_t result = 1;
    if (k > n) {
        return 0;
    }
    if (k > n - k) {
        k = n - k;
    }
    for (unsigned int i = 1; i <= k; i++) {
        result = (result * (uint64_t)(n - k + i)) / (uint64_t)i;
    }
    return result;
}

static int
max_probabilities(unsigned int ploidy, unsigned int n_alleles, int phased, uint32_t *value)
{
    uint64_t width;
    if (n_alleles == 0) {
        PyErr_SetString(PyExc_ValueError, "BGEN allele count must be positive.");
        return -1;
    }
    if (phased) {
        width = (uint64_t)ploidy * (uint64_t)n_alleles;
    } else {
        width = choose_u64(ploidy + n_alleles - 1, n_alleles - 1);
    }
    if (width > UINT32_MAX) {
        PyErr_SetString(PyExc_ValueError, "BGEN probability width is too large.");
        return -1;
    }
    *value = (uint32_t)width;
    return 0;
}

static uint32_t
read_bits_lsb(const unsigned char *data, Py_ssize_t data_len, uint64_t bit_offset, unsigned int bit_depth, int *ok)
{
    uint64_t byte_offset = bit_offset >> 3;
    unsigned int shift = (unsigned int)(bit_offset & 7U);
    uint64_t value = 0;

    if (bit_depth == 8 && shift == 0) {
        if (byte_offset >= (uint64_t)data_len) {
            *ok = 0;
            return 0;
        }
        return data[byte_offset];
    }
    if (bit_depth == 16 && shift == 0) {
        if (byte_offset + 2 > (uint64_t)data_len) {
            *ok = 0;
            return 0;
        }
        return (uint32_t)data[byte_offset] | ((uint32_t)data[byte_offset + 1] << 8);
    }
    if (bit_depth == 32 && shift == 0) {
        if (byte_offset + 4 > (uint64_t)data_len) {
            *ok = 0;
            return 0;
        }
        return read_u32_le(data + byte_offset);
    }

    if (bit_depth == 0 || bit_depth > 32 || bit_offset + bit_depth > (uint64_t)data_len * 8U) {
        *ok = 0;
        return 0;
    }
    for (unsigned int i = 0; i < bit_depth; i++) {
        uint64_t bit = bit_offset + i;
        if (data[bit >> 3] & (1U << (bit & 7U))) {
            value |= (uint64_t)1 << i;
        }
    }
    return (uint32_t)value;
}

static void
write_bits_lsb(unsigned char *data, uint64_t bit_offset, unsigned int bit_depth, uint32_t value)
{
    if (bit_depth == 8 && (bit_offset & 7U) == 0) {
        data[bit_offset >> 3] = (unsigned char)value;
        return;
    }
    if (bit_depth == 16 && (bit_offset & 7U) == 0) {
        uint64_t byte_offset = bit_offset >> 3;
        data[byte_offset] = (unsigned char)(value & 0xFF);
        data[byte_offset + 1] = (unsigned char)((value >> 8) & 0xFF);
        return;
    }
    if (bit_depth == 32 && (bit_offset & 7U) == 0) {
        write_u32_le(data + (bit_offset >> 3), value);
        return;
    }
    for (unsigned int i = 0; i < bit_depth; i++) {
        uint64_t bit = bit_offset + i;
        unsigned char mask = (unsigned char)(1U << (bit & 7U));
        if (value & ((uint32_t)1 << i)) {
            data[bit >> 3] |= mask;
        } else {
            data[bit >> 3] &= (unsigned char)~mask;
        }
    }
}

static void
fill_nan(float *out, uint32_t width)
{
    for (uint32_t i = 0; i < width; i++) {
        out[i] = NAN;
    }
}

typedef struct {
    uint32_t first_variant_offset;
    uint32_t n_variants;
    uint32_t n_samples;
    uint32_t compression;
    uint32_t layout;
} BgenHeader;

typedef struct {
    uint32_t n_samples;
    uint16_t n_alleles;
    uint8_t min_ploidy;
    uint8_t max_ploidy;
    const unsigned char *ploidy_bytes;
    int phased;
    uint8_t bit_depth;
    uint32_t width;
    const unsigned char *probabilities;
    Py_ssize_t probabilities_len;
} Layout2Info;

static int
file_read_exact(FILE *fp, void *buffer, size_t size, const char *context)
{
    if (size == 0) {
        return 0;
    }
    if (fread(buffer, 1, size, fp) != size) {
        PyErr_Format(PyExc_ValueError, "Malformed BGEN file: %s is truncated.", context);
        return -1;
    }
    return 0;
}

static int
file_read_u16(FILE *fp, uint16_t *value, const char *context)
{
    unsigned char buffer[2];
    if (file_read_exact(fp, buffer, sizeof(buffer), context) < 0) {
        return -1;
    }
    *value = read_u16_le(buffer);
    return 0;
}

static int
file_read_u32(FILE *fp, uint32_t *value, const char *context)
{
    unsigned char buffer[4];
    if (file_read_exact(fp, buffer, sizeof(buffer), context) < 0) {
        return -1;
    }
    *value = read_u32_le(buffer);
    return 0;
}

static int
file_skip(FILE *fp, uint64_t size, const char *context)
{
    while (size > 0) {
        long step = size > (uint64_t)LONG_MAX ? LONG_MAX : (long)size;
        if (fseek(fp, step, SEEK_CUR) != 0) {
            PyErr_Format(PyExc_ValueError, "Malformed BGEN file: could not skip %s.", context);
            return -1;
        }
        size -= (uint64_t)step;
    }
    return 0;
}

static int
file_skip_len_prefixed_text(FILE *fp, int len_size, const char *context)
{
    uint32_t size;
    if (len_size == 2) {
        uint16_t small_size;
        if (file_read_u16(fp, &small_size, context) < 0) {
            return -1;
        }
        size = (uint32_t)small_size;
    } else {
        if (file_read_u32(fp, &size, context) < 0) {
            return -1;
        }
    }
    return file_skip(fp, size, context);
}

static int
ensure_byte_capacity(unsigned char **buffer, size_t *capacity, size_t size)
{
    unsigned char *tmp;
    if (size <= *capacity) {
        return 0;
    }
    tmp = (unsigned char *)PyMem_Realloc(*buffer, size == 0 ? 1 : size);
    if (tmp == NULL) {
        PyErr_NoMemory();
        return -1;
    }
    *buffer = tmp;
    *capacity = size;
    return 0;
}

static PyObject *
allocate_numpy_float32_array(int ndim, const Py_ssize_t *dims, Py_buffer *view)
{
    PyObject *numpy = NULL;
    PyObject *empty = NULL;
    PyObject *dtype = NULL;
    PyObject *shape = NULL;
    PyObject *args = NULL;
    PyObject *kwargs = NULL;
    PyObject *array = NULL;

    numpy = PyImport_ImportModule("numpy");
    if (numpy == NULL) {
        goto error;
    }
    empty = PyObject_GetAttrString(numpy, "empty");
    dtype = PyObject_GetAttrString(numpy, "float32");
    if (empty == NULL || dtype == NULL) {
        goto error;
    }
    shape = PyTuple_New(ndim);
    if (shape == NULL) {
        goto error;
    }
    for (int i = 0; i < ndim; i++) {
        PyObject *dim = PyLong_FromSsize_t(dims[i]);
        if (dim == NULL) {
            goto error;
        }
        PyTuple_SET_ITEM(shape, i, dim);
    }
    args = PyTuple_Pack(1, shape);
    kwargs = PyDict_New();
    if (args == NULL || kwargs == NULL || PyDict_SetItemString(kwargs, "dtype", dtype) < 0) {
        goto error;
    }
    array = PyObject_Call(empty, args, kwargs);
    if (array == NULL) {
        goto error;
    }
    if (PyObject_GetBuffer(array, view, PyBUF_WRITABLE) < 0) {
        goto error;
    }
    if (view->itemsize != (Py_ssize_t)sizeof(float)) {
        PyBuffer_Release(view);
        PyErr_SetString(PyExc_TypeError, "NumPy did not allocate a float32 output array.");
        goto error;
    }

    Py_DECREF(kwargs);
    Py_DECREF(args);
    Py_DECREF(shape);
    Py_DECREF(dtype);
    Py_DECREF(empty);
    Py_DECREF(numpy);
    return array;

error:
    Py_XDECREF(array);
    Py_XDECREF(kwargs);
    Py_XDECREF(args);
    Py_XDECREF(shape);
    Py_XDECREF(dtype);
    Py_XDECREF(empty);
    Py_XDECREF(numpy);
    return NULL;
}

static int
read_bgen_header(FILE *fp, BgenHeader *header)
{
    unsigned char buffer[20];
    uint32_t offset;
    uint32_t header_length;
    uint32_t flags;

    if (file_read_exact(fp, buffer, sizeof(buffer), "header") < 0) {
        return -1;
    }
    offset = read_u32_le(buffer);
    header_length = read_u32_le(buffer + 4);
    header->n_variants = read_u32_le(buffer + 8);
    header->n_samples = read_u32_le(buffer + 12);
    if (memcmp(buffer + 16, "bgen", 4) != 0 && read_u32_le(buffer + 16) != 0) {
        PyErr_SetString(PyExc_ValueError, "File does not appear to be a BGEN file.");
        return -1;
    }
    if (header_length < 20) {
        PyErr_SetString(PyExc_ValueError, "Malformed BGEN file: header length is too small.");
        return -1;
    }
    if (file_skip(fp, (uint64_t)(header_length - 20), "free data") < 0) {
        return -1;
    }
    if (file_read_u32(fp, &flags, "flags") < 0) {
        return -1;
    }
    header->first_variant_offset = offset + 4U;
    header->compression = flags & 0x3U;
    header->layout = (flags >> 2) & 0xFU;
    if (fseek(fp, (long)header->first_variant_offset, SEEK_SET) != 0) {
        PyErr_SetString(PyExc_ValueError, "Malformed BGEN file: could not seek to the first variant.");
        return -1;
    }
    return 0;
}

static int
read_variant_prefix(FILE *fp, uint16_t *n_alleles, uint32_t *block_len)
{
    uint32_t pos;
    if (file_skip_len_prefixed_text(fp, 2, "variant ID") < 0
            || file_skip_len_prefixed_text(fp, 2, "RSID") < 0
            || file_skip_len_prefixed_text(fp, 2, "chromosome") < 0
            || file_read_u32(fp, &pos, "position") < 0
            || file_read_u16(fp, n_alleles, "allele count") < 0) {
        return -1;
    }
    (void)pos;
    for (uint16_t allele = 0; allele < *n_alleles; allele++) {
        if (file_skip_len_prefixed_text(fp, 4, "allele") < 0) {
            return -1;
        }
    }
    return file_read_u32(fp, block_len, "genotype block length");
}

static int
read_variant_payload(
    FILE *fp,
    uint32_t compression,
    uint32_t block_len,
    unsigned char **block_buffer,
    size_t *block_capacity,
    unsigned char **payload_buffer,
    size_t *payload_capacity,
    const unsigned char **payload,
    Py_ssize_t *payload_len)
{
    if (block_len > (uint32_t)PY_SSIZE_T_MAX) {
        PyErr_SetString(PyExc_MemoryError, "BGEN genotype block is too large.");
        return -1;
    }
    if (ensure_byte_capacity(block_buffer, block_capacity, (size_t)block_len) < 0) {
        return -1;
    }
    if (file_read_exact(fp, *block_buffer, (size_t)block_len, "genotype block") < 0) {
        return -1;
    }

    if (compression == 0) {
        *payload = *block_buffer;
        *payload_len = (Py_ssize_t)block_len;
        return 0;
    }

    if (block_len < 4) {
        PyErr_SetString(PyExc_ValueError, "Malformed BGEN genotype block: compressed length field is truncated.");
        return -1;
    }
    if (compression == 2) {
        PyErr_SetString(
            PyExc_NotImplementedError,
            "Native bulk BGEN reading currently supports uncompressed and zlib-compressed files.");
        return -1;
    }
    if (compression != 1) {
        PyErr_SetString(PyExc_ValueError, "Unsupported BGEN compression flag.");
        return -1;
    }

    {
        uint32_t expected_len = read_u32_le(*block_buffer);
        uLongf dest_len = (uLongf)expected_len;
        int zlib_status;
        if (expected_len > (uint32_t)PY_SSIZE_T_MAX) {
            PyErr_SetString(PyExc_MemoryError, "BGEN decompressed genotype block is too large.");
            return -1;
        }
        if (ensure_byte_capacity(payload_buffer, payload_capacity, (size_t)expected_len) < 0) {
            return -1;
        }
        zlib_status = uncompress(
            *payload_buffer,
            &dest_len,
            *block_buffer + 4,
            (uLong)(block_len - 4U));
        if (zlib_status != Z_OK || dest_len != (uLongf)expected_len) {
            PyErr_SetString(PyExc_ValueError, "BGEN genotype block decompressed to the wrong size.");
            return -1;
        }
        *payload = *payload_buffer;
        *payload_len = (Py_ssize_t)expected_len;
    }
    return 0;
}

static int
parse_layout2_info(
    const unsigned char *data,
    Py_ssize_t data_len,
    uint32_t expected_samples,
    uint16_t expected_alleles,
    Layout2Info *info)
{
    Py_ssize_t prob_offset;
    if (data_len < 10) {
        PyErr_SetString(PyExc_ValueError, "Malformed BGEN genotype block: layout-2 header is truncated.");
        return -1;
    }
    info->n_samples = read_u32_le(data);
    info->n_alleles = read_u16_le(data + 4);
    if (info->n_samples != expected_samples || info->n_alleles != expected_alleles) {
        PyErr_SetString(PyExc_ValueError, "Malformed BGEN genotype block: sample or allele count mismatch.");
        return -1;
    }
    info->min_ploidy = data[6];
    info->max_ploidy = data[7];
    if (8 + (Py_ssize_t)info->n_samples + 2 > data_len) {
        PyErr_SetString(PyExc_ValueError, "Malformed BGEN genotype block: ploidy bytes are truncated.");
        return -1;
    }
    info->ploidy_bytes = data + 8;
    info->phased = (int)data[8 + info->n_samples];
    info->bit_depth = data[8 + info->n_samples + 1];
    if (info->bit_depth < 1 || info->bit_depth > 32) {
        PyErr_SetString(PyExc_ValueError, "BGEN probability bit depth must be between 1 and 32.");
        return -1;
    }
    if (max_probabilities(info->max_ploidy, info->n_alleles, info->phased, &info->width) < 0) {
        return -1;
    }
    if (info->width == 0) {
        PyErr_SetString(PyExc_ValueError, "BGEN probability width cannot be zero.");
        return -1;
    }
    prob_offset = 10 + (Py_ssize_t)info->n_samples;
    info->probabilities = data + prob_offset;
    info->probabilities_len = data_len - prob_offset;
    return 0;
}

static double
bit_depth_denominator(uint8_t bit_depth)
{
    return bit_depth == 32 ? 4294967295.0 : (double)(((uint64_t)1 << bit_depth) - 1U);
}

#if BGEN_HAVE_AVX2
static int
bgen_cpu_has_avx2(void)
{
    static int checked = 0;
    static int has_avx2 = 0;
    if (!checked) {
        __builtin_cpu_init();
        has_avx2 = __builtin_cpu_supports("avx2") ? 1 : 0;
        checked = 1;
    }
    return has_avx2;
}

__attribute__((target("avx2")))
static int
ploidy_all_value_avx2(const unsigned char *ploidy, uint32_t n_samples, unsigned char value)
{
    uint32_t sample = 0;
    __m256i expected = _mm256_set1_epi8((char)value);
    for (; sample + 32 <= n_samples; sample += 32) {
        __m256i observed = _mm256_loadu_si256((const __m256i *)(ploidy + sample));
        __m256i equal = _mm256_cmpeq_epi8(observed, expected);
        if ((uint32_t)_mm256_movemask_epi8(equal) != 0xFFFFFFFFU) {
            return 0;
        }
    }
    for (; sample < n_samples; sample++) {
        if (ploidy[sample] != value) {
            return 0;
        }
    }
    return 1;
}

__attribute__((target("avx2")))
static void
store_interleaved_prob_pairs_avx2(float *out, __m256 probs, __m256 complements)
{
    __m256 lo = _mm256_unpacklo_ps(probs, complements);
    __m256 hi = _mm256_unpackhi_ps(probs, complements);
    _mm256_storeu_ps(out, _mm256_permute2f128_ps(lo, hi, 0x20));
    _mm256_storeu_ps(out + 8, _mm256_permute2f128_ps(lo, hi, 0x31));
}

__attribute__((target("avx2")))
static void
decode_phased_biallelic16_probabilities_avx2(const unsigned char *bits, uint32_t n_samples, float *out)
{
    uint32_t sample = 0;
    const __m256 scale = _mm256_set1_ps(1.0f / 65535.0f);
    const __m256 one = _mm256_set1_ps(1.0f);

    for (; sample + 8 <= n_samples; sample += 8) {
        __m256i packed = _mm256_loadu_si256((const __m256i *)(bits + (uint64_t)sample * 4U));
        __m128i low_words = _mm256_castsi256_si128(packed);
        __m128i high_words = _mm256_extracti128_si256(packed, 1);
        __m256 low_probs = _mm256_mul_ps(_mm256_cvtepi32_ps(_mm256_cvtepu16_epi32(low_words)), scale);
        __m256 high_probs = _mm256_mul_ps(_mm256_cvtepi32_ps(_mm256_cvtepu16_epi32(high_words)), scale);

        store_interleaved_prob_pairs_avx2(out + (uint64_t)sample * 4U, low_probs, _mm256_sub_ps(one, low_probs));
        store_interleaved_prob_pairs_avx2(out + (uint64_t)sample * 4U + 16U, high_probs, _mm256_sub_ps(one, high_probs));
    }

    for (; sample < n_samples; sample++) {
        const unsigned char *ptr = bits + (uint64_t)sample * 4U;
        float ref0 = (float)read_u16_le(ptr) * (1.0f / 65535.0f);
        float ref1 = (float)read_u16_le(ptr + 2) * (1.0f / 65535.0f);
        float *row = out + (uint64_t)sample * 4U;
        row[0] = ref0;
        row[1] = 1.0f - ref0;
        row[2] = ref1;
        row[3] = 1.0f - ref1;
    }
}
#endif

static int
decode_layout2_probabilities_into(
    const unsigned char *data,
    Py_ssize_t data_len,
    uint32_t expected_samples,
    uint16_t expected_alleles,
    uint32_t output_width,
    float *out)
{
    Layout2Info info;
    uint64_t bit_offset = 0;
    double denominator;

    if (parse_layout2_info(data, data_len, expected_samples, expected_alleles, &info) < 0) {
        return -1;
    }
    if (output_width != 0 && info.width > output_width) {
        PyErr_SetString(PyExc_ValueError, "Native bulk BGEN probability reader encountered a probability width larger than the allocated output width.");
        return -1;
    }

#if BGEN_HAVE_AVX2
    if (output_width == 4 && info.n_alleles == 2 && info.min_ploidy == 2 && info.max_ploidy == 2 && info.phased && info.bit_depth == 16) {
        if ((uint64_t)info.probabilities_len < (uint64_t)info.n_samples * 4U) {
            PyErr_SetString(PyExc_ValueError, "Malformed BGEN genotype block: probability bits are truncated.");
            return -1;
        }
        if (bgen_cpu_has_avx2() && ploidy_all_value_avx2(info.ploidy_bytes, info.n_samples, 2)) {
            decode_phased_biallelic16_probabilities_avx2(info.probabilities, info.n_samples, out);
            return 0;
        }
    }
#endif

    if (info.n_alleles == 2 && info.min_ploidy == 2 && info.max_ploidy == 2 && info.phased && info.bit_depth == 16) {
        const unsigned char *bits = info.probabilities;
        if ((uint64_t)info.probabilities_len < (uint64_t)info.n_samples * 4U) {
            PyErr_SetString(PyExc_ValueError, "Malformed BGEN genotype block: probability bits are truncated.");
            return -1;
        }
        for (uint32_t sample = 0; sample < info.n_samples; sample++) {
            uint32_t raw0 = read_u16_le(bits);
            uint32_t raw1 = read_u16_le(bits + 2);
            int missing = (info.ploidy_bytes[sample] & 0x80U) != 0;
            float *row = out + (uint64_t)sample * (uint64_t)output_width;
            bits += 4;
            if (missing) {
                fill_nan(row, output_width);
            } else {
                float ref0 = (float)raw0 * (1.0f / 65535.0f);
                float ref1 = (float)raw1 * (1.0f / 65535.0f);
                row[0] = ref0;
                row[1] = 1.0f - ref0;
                row[2] = ref1;
                row[3] = 1.0f - ref1;
                for (uint32_t j = 4; j < output_width; j++) {
                    row[j] = NAN;
                }
            }
        }
        return 0;
    }

    if (info.n_alleles == 2 && info.min_ploidy == 2 && info.max_ploidy == 2 && !info.phased && info.bit_depth == 16) {
        const unsigned char *bits = info.probabilities;
        if ((uint64_t)info.probabilities_len < (uint64_t)info.n_samples * 4U) {
            PyErr_SetString(PyExc_ValueError, "Malformed BGEN genotype block: probability bits are truncated.");
            return -1;
        }
        for (uint32_t sample = 0; sample < info.n_samples; sample++) {
            uint32_t raw0 = read_u16_le(bits);
            uint32_t raw1 = read_u16_le(bits + 2);
            int missing = (info.ploidy_bytes[sample] & 0x80U) != 0;
            float *row = out + (uint64_t)sample * (uint64_t)output_width;
            bits += 4;
            if (missing) {
                fill_nan(row, output_width);
            } else {
                float p0 = (float)raw0 * (1.0f / 65535.0f);
                float p1 = (float)raw1 * (1.0f / 65535.0f);
                row[0] = p0;
                row[1] = p1;
                row[2] = 1.0f - p0 - p1;
                for (uint32_t j = 3; j < output_width; j++) {
                    row[j] = NAN;
                }
            }
        }
        return 0;
    }

    if (info.n_alleles == 2 && info.min_ploidy == 2 && info.max_ploidy == 2 && info.phased && info.bit_depth == 8) {
        const unsigned char *bits = info.probabilities;
        if ((uint64_t)info.probabilities_len < (uint64_t)info.n_samples * 2U) {
            PyErr_SetString(PyExc_ValueError, "Malformed BGEN genotype block: probability bits are truncated.");
            return -1;
        }
        for (uint32_t sample = 0; sample < info.n_samples; sample++) {
            uint32_t raw0 = bits[0];
            uint32_t raw1 = bits[1];
            int missing = (info.ploidy_bytes[sample] & 0x80U) != 0;
            float *row = out + (uint64_t)sample * (uint64_t)output_width;
            bits += 2;
            if (missing) {
                fill_nan(row, output_width);
            } else {
                float ref0 = (float)raw0 * (1.0f / 255.0f);
                float ref1 = (float)raw1 * (1.0f / 255.0f);
                row[0] = ref0;
                row[1] = 1.0f - ref0;
                row[2] = ref1;
                row[3] = 1.0f - ref1;
                for (uint32_t j = 4; j < output_width; j++) {
                    row[j] = NAN;
                }
            }
        }
        return 0;
    }

    if (info.n_alleles == 2 && info.min_ploidy == 2 && info.max_ploidy == 2 && !info.phased && info.bit_depth == 8) {
        const unsigned char *bits = info.probabilities;
        if ((uint64_t)info.probabilities_len < (uint64_t)info.n_samples * 2U) {
            PyErr_SetString(PyExc_ValueError, "Malformed BGEN genotype block: probability bits are truncated.");
            return -1;
        }
        for (uint32_t sample = 0; sample < info.n_samples; sample++) {
            uint32_t raw0 = bits[0];
            uint32_t raw1 = bits[1];
            int missing = (info.ploidy_bytes[sample] & 0x80U) != 0;
            float *row = out + (uint64_t)sample * (uint64_t)output_width;
            bits += 2;
            if (missing) {
                fill_nan(row, output_width);
            } else {
                float p0 = (float)raw0 * (1.0f / 255.0f);
                float p1 = (float)raw1 * (1.0f / 255.0f);
                row[0] = p0;
                row[1] = p1;
                row[2] = 1.0f - p0 - p1;
                for (uint32_t j = 3; j < output_width; j++) {
                    row[j] = NAN;
                }
            }
        }
        return 0;
    }

    denominator = bit_depth_denominator(info.bit_depth);
    for (uint32_t sample = 0; sample < info.n_samples; sample++) {
        uint8_t ploidy_byte = info.ploidy_bytes[sample];
        uint32_t ploidy = (uint32_t)(ploidy_byte & 0x3FU);
        int missing = (ploidy_byte & 0x80U) != 0;
        float *row = out + ((uint64_t)sample * (uint64_t)output_width);

        if (ploidy < info.min_ploidy || ploidy > info.max_ploidy) {
            PyErr_SetString(PyExc_ValueError, "Malformed BGEN genotype block: sample ploidy is out of range.");
            return -1;
        }

        fill_nan(row, output_width);
        if (!info.phased) {
            uint32_t sample_width;
            uint32_t stored;
            double remainder = 1.0;
            if (max_probabilities(ploidy, info.n_alleles, 0, &sample_width) < 0) {
                return -1;
            }
            stored = sample_width > 0 ? sample_width - 1 : 0;
            for (uint32_t j = 0; j < stored; j++) {
                int ok = 1;
                uint32_t raw = read_bits_lsb(info.probabilities, info.probabilities_len, bit_offset, info.bit_depth, &ok);
                if (!ok) {
                    PyErr_SetString(PyExc_ValueError, "Malformed BGEN genotype block: probability bits are truncated.");
                    return -1;
                }
                bit_offset += info.bit_depth;
                if (!missing) {
                    double prob = (double)raw / denominator;
                    row[j] = (float)prob;
                    remainder -= prob;
                }
            }
            if (!missing && sample_width > 0) {
                row[stored] = (float)remainder;
            }
        } else {
            for (uint32_t hap = 0; hap < ploidy; hap++) {
                double remainder = 1.0;
                for (uint32_t allele = 0; allele + 1 < info.n_alleles; allele++) {
                    int ok = 1;
                    uint32_t raw = read_bits_lsb(info.probabilities, info.probabilities_len, bit_offset, info.bit_depth, &ok);
                    if (!ok) {
                        PyErr_SetString(PyExc_ValueError, "Malformed BGEN genotype block: probability bits are truncated.");
                        return -1;
                    }
                    bit_offset += info.bit_depth;
                    if (!missing) {
                        double prob = (double)raw / denominator;
                        row[hap * info.n_alleles + allele] = (float)prob;
                        remainder -= prob;
                    }
                }
                if (!missing) {
                    row[hap * info.n_alleles + (info.n_alleles - 1)] = (float)remainder;
                }
            }
        }
    }
    return 0;
}

static int
decode_layout2_dosage_into(
    const unsigned char *data,
    Py_ssize_t data_len,
    uint32_t expected_samples,
    uint16_t expected_alleles,
    float *out)
{
    Layout2Info info;
    uint64_t bit_offset = 0;
    double denominator;

    if (parse_layout2_info(data, data_len, expected_samples, expected_alleles, &info) < 0) {
        return -1;
    }
    if (info.n_alleles != 2) {
        PyErr_SetString(PyExc_ValueError, "Native BGEN dosage reading currently supports biallelic variants.");
        return -1;
    }

    if (info.min_ploidy == 2 && info.max_ploidy == 2 && info.phased && info.bit_depth == 16) {
        const unsigned char *bits = info.probabilities;
        if ((uint64_t)info.probabilities_len < (uint64_t)info.n_samples * 4U) {
            PyErr_SetString(PyExc_ValueError, "Malformed BGEN genotype block: probability bits are truncated.");
            return -1;
        }
        for (uint32_t sample = 0; sample < info.n_samples; sample++) {
            uint32_t raw0 = read_u16_le(bits);
            uint32_t raw1 = read_u16_le(bits + 2);
            bits += 4;
            if (info.ploidy_bytes[sample] & 0x80U) {
                out[sample] = NAN;
            } else {
                out[sample] = 2.0f - ((float)(raw0 + raw1) * (1.0f / 65535.0f));
            }
        }
        return 0;
    }

    if (info.min_ploidy == 2 && info.max_ploidy == 2 && !info.phased && info.bit_depth == 16) {
        const unsigned char *bits = info.probabilities;
        if ((uint64_t)info.probabilities_len < (uint64_t)info.n_samples * 4U) {
            PyErr_SetString(PyExc_ValueError, "Malformed BGEN genotype block: probability bits are truncated.");
            return -1;
        }
        for (uint32_t sample = 0; sample < info.n_samples; sample++) {
            uint32_t raw0 = read_u16_le(bits);
            uint32_t raw1 = read_u16_le(bits + 2);
            bits += 4;
            if (info.ploidy_bytes[sample] & 0x80U) {
                out[sample] = NAN;
            } else {
                float p0 = (float)raw0 * (1.0f / 65535.0f);
                float p1 = (float)raw1 * (1.0f / 65535.0f);
                out[sample] = 2.0f - 2.0f * p0 - p1;
            }
        }
        return 0;
    }

    if (info.min_ploidy == 2 && info.max_ploidy == 2 && info.phased && info.bit_depth == 8) {
        const unsigned char *bits = info.probabilities;
        if ((uint64_t)info.probabilities_len < (uint64_t)info.n_samples * 2U) {
            PyErr_SetString(PyExc_ValueError, "Malformed BGEN genotype block: probability bits are truncated.");
            return -1;
        }
        for (uint32_t sample = 0; sample < info.n_samples; sample++) {
            uint32_t raw0 = bits[0];
            uint32_t raw1 = bits[1];
            bits += 2;
            if (info.ploidy_bytes[sample] & 0x80U) {
                out[sample] = NAN;
            } else {
                out[sample] = 2.0f - ((float)(raw0 + raw1) * (1.0f / 255.0f));
            }
        }
        return 0;
    }

    if (info.min_ploidy == 2 && info.max_ploidy == 2 && !info.phased && info.bit_depth == 8) {
        const unsigned char *bits = info.probabilities;
        if ((uint64_t)info.probabilities_len < (uint64_t)info.n_samples * 2U) {
            PyErr_SetString(PyExc_ValueError, "Malformed BGEN genotype block: probability bits are truncated.");
            return -1;
        }
        for (uint32_t sample = 0; sample < info.n_samples; sample++) {
            uint32_t raw0 = bits[0];
            uint32_t raw1 = bits[1];
            bits += 2;
            if (info.ploidy_bytes[sample] & 0x80U) {
                out[sample] = NAN;
            } else {
                float p0 = (float)raw0 * (1.0f / 255.0f);
                float p1 = (float)raw1 * (1.0f / 255.0f);
                out[sample] = 2.0f - 2.0f * p0 - p1;
            }
        }
        return 0;
    }

    denominator = bit_depth_denominator(info.bit_depth);
    for (uint32_t sample = 0; sample < info.n_samples; sample++) {
        uint8_t ploidy_byte = info.ploidy_bytes[sample];
        uint32_t ploidy = (uint32_t)(ploidy_byte & 0x3FU);
        int missing = (ploidy_byte & 0x80U) != 0;
        double dosage = 0.0;

        if (ploidy < info.min_ploidy || ploidy > info.max_ploidy) {
            PyErr_SetString(PyExc_ValueError, "Malformed BGEN genotype block: sample ploidy is out of range.");
            return -1;
        }

        if (!info.phased) {
            uint32_t sample_width;
            uint32_t stored;
            double remainder = 1.0;
            if (max_probabilities(ploidy, info.n_alleles, 0, &sample_width) < 0) {
                return -1;
            }
            stored = sample_width > 0 ? sample_width - 1 : 0;
            for (uint32_t j = 0; j < stored; j++) {
                int ok = 1;
                uint32_t raw = read_bits_lsb(info.probabilities, info.probabilities_len, bit_offset, info.bit_depth, &ok);
                if (!ok) {
                    PyErr_SetString(PyExc_ValueError, "Malformed BGEN genotype block: probability bits are truncated.");
                    return -1;
                }
                bit_offset += info.bit_depth;
                if (!missing) {
                    double prob = (double)raw / denominator;
                    dosage += (double)j * prob;
                    remainder -= prob;
                }
            }
            if (!missing) {
                dosage += (double)stored * remainder;
                out[sample] = (float)dosage;
            } else {
                out[sample] = NAN;
            }
        } else {
            for (uint32_t hap = 0; hap < ploidy; hap++) {
                int ok = 1;
                uint32_t raw = read_bits_lsb(info.probabilities, info.probabilities_len, bit_offset, info.bit_depth, &ok);
                if (!ok) {
                    PyErr_SetString(PyExc_ValueError, "Malformed BGEN genotype block: probability bits are truncated.");
                    return -1;
                }
                bit_offset += info.bit_depth;
                if (!missing) {
                    dosage += 1.0 - ((double)raw / denominator);
                }
            }
            out[sample] = missing ? NAN : (float)dosage;
        }
    }
    return 0;
}

static PyObject *
read_file_probabilities(PyObject *self, PyObject *args)
{
    const char *path;
    FILE *fp = NULL;
    BgenHeader header;
    unsigned char *block_buffer = NULL;
    unsigned char *payload_buffer = NULL;
    size_t block_capacity = 0;
    size_t payload_capacity = 0;
    PyObject *out_obj = NULL;
    Py_buffer out_view;
    int have_out_view = 0;
    float *out = NULL;
    uint32_t out_width = 0;

    if (!PyArg_ParseTuple(args, "s", &path)) {
        return NULL;
    }
    fp = fopen(path, "rb");
    if (fp == NULL) {
        PyErr_SetFromErrnoWithFilename(PyExc_OSError, path);
        return NULL;
    }
    if (read_bgen_header(fp, &header) < 0) {
        goto error;
    }
    if (header.layout != 2) {
        PyErr_SetString(PyExc_NotImplementedError, "Native bulk BGEN reading currently supports layout-2 files.");
        goto error;
    }
    if (header.compression != 0 && header.compression != 1 && header.compression != 2) {
        PyErr_SetString(PyExc_ValueError, "Unsupported BGEN compression flag.");
        goto error;
    }

    if (header.n_variants == 0) {
        Py_ssize_t dims[3];
        dims[0] = 0;
        dims[1] = (Py_ssize_t)header.n_samples;
        dims[2] = 0;
        out_obj = allocate_numpy_float32_array(3, dims, &out_view);
        if (out_obj == NULL) {
            goto error;
        }
        have_out_view = 1;
    }

    for (uint32_t variant = 0; variant < header.n_variants; variant++) {
        uint16_t n_alleles;
        uint32_t block_len;
        const unsigned char *payload;
        Py_ssize_t payload_len;

        if (read_variant_prefix(fp, &n_alleles, &block_len) < 0
                || read_variant_payload(
                    fp,
                    header.compression,
                    block_len,
                    &block_buffer,
                    &block_capacity,
                    &payload_buffer,
                    &payload_capacity,
                    &payload,
                    &payload_len) < 0) {
            goto error;
        }

        if (variant == 0) {
            Layout2Info info;
            Py_ssize_t dims[3];
            if (parse_layout2_info(payload, payload_len, header.n_samples, n_alleles, &info) < 0) {
                goto error;
            }
            out_width = info.width;
            dims[0] = (Py_ssize_t)header.n_variants;
            dims[1] = (Py_ssize_t)header.n_samples;
            dims[2] = (Py_ssize_t)out_width;
            out_obj = allocate_numpy_float32_array(3, dims, &out_view);
            if (out_obj == NULL) {
                goto error;
            }
            have_out_view = 1;
            out = (float *)out_view.buf;
        }

        if (decode_layout2_probabilities_into(
                payload,
                payload_len,
                header.n_samples,
                n_alleles,
                out_width,
                out + (uint64_t)variant * (uint64_t)header.n_samples * (uint64_t)out_width) < 0) {
            goto error;
        }
    }

    if (have_out_view) {
        PyBuffer_Release(&out_view);
        have_out_view = 0;
    }
    PyMem_Free(block_buffer);
    PyMem_Free(payload_buffer);
    fclose(fp);
    return Py_BuildValue("NIII", out_obj, header.n_variants, header.n_samples, out_width);

error:
    if (have_out_view) {
        PyBuffer_Release(&out_view);
    }
    Py_XDECREF(out_obj);
    PyMem_Free(block_buffer);
    PyMem_Free(payload_buffer);
    if (fp != NULL) {
        fclose(fp);
    }
    return NULL;
}

static PyObject *
read_file_dosage(PyObject *self, PyObject *args)
{
    const char *path;
    FILE *fp = NULL;
    BgenHeader header;
    unsigned char *block_buffer = NULL;
    unsigned char *payload_buffer = NULL;
    size_t block_capacity = 0;
    size_t payload_capacity = 0;
    PyObject *out_obj = NULL;
    Py_buffer out_view;
    int have_out_view = 0;
    float *out = NULL;
    Py_ssize_t dims[2];

    if (!PyArg_ParseTuple(args, "s", &path)) {
        return NULL;
    }
    fp = fopen(path, "rb");
    if (fp == NULL) {
        PyErr_SetFromErrnoWithFilename(PyExc_OSError, path);
        return NULL;
    }
    if (read_bgen_header(fp, &header) < 0) {
        goto error;
    }
    if (header.layout != 2) {
        PyErr_SetString(PyExc_NotImplementedError, "Native bulk BGEN reading currently supports layout-2 files.");
        goto error;
    }
    if (header.compression != 0 && header.compression != 1 && header.compression != 2) {
        PyErr_SetString(PyExc_ValueError, "Unsupported BGEN compression flag.");
        goto error;
    }
    dims[0] = (Py_ssize_t)header.n_variants;
    dims[1] = (Py_ssize_t)header.n_samples;
    out_obj = allocate_numpy_float32_array(2, dims, &out_view);
    if (out_obj == NULL) {
        goto error;
    }
    have_out_view = 1;
    out = (float *)out_view.buf;

    for (uint32_t variant = 0; variant < header.n_variants; variant++) {
        uint16_t n_alleles;
        uint32_t block_len;
        const unsigned char *payload;
        Py_ssize_t payload_len;

        if (read_variant_prefix(fp, &n_alleles, &block_len) < 0
                || read_variant_payload(
                    fp,
                    header.compression,
                    block_len,
                    &block_buffer,
                    &block_capacity,
                    &payload_buffer,
                    &payload_capacity,
                    &payload,
                    &payload_len) < 0) {
            goto error;
        }
        if (decode_layout2_dosage_into(
                payload,
                payload_len,
                header.n_samples,
                n_alleles,
                out + (uint64_t)variant * (uint64_t)header.n_samples) < 0) {
            goto error;
        }
    }

    if (have_out_view) {
        PyBuffer_Release(&out_view);
        have_out_view = 0;
    }
    PyMem_Free(block_buffer);
    PyMem_Free(payload_buffer);
    fclose(fp);
    return Py_BuildValue("NII", out_obj, header.n_variants, header.n_samples);

error:
    if (have_out_view) {
        PyBuffer_Release(&out_view);
    }
    Py_XDECREF(out_obj);
    PyMem_Free(block_buffer);
    PyMem_Free(payload_buffer);
    if (fp != NULL) {
        fclose(fp);
    }
    return NULL;
}

static PyObject *
decode_layout2(PyObject *self, PyObject *args)
{
    Py_buffer payload;
    unsigned int expected_samples;
    unsigned int expected_alleles;
    const unsigned char *data;
    Py_ssize_t data_len;
    uint32_t n_samples;
    uint16_t n_alleles;
    uint8_t min_ploidy;
    uint8_t max_ploidy;
    const unsigned char *ploidy_bytes;
    int phased;
    uint8_t bit_depth;
    uint32_t width;
    double denominator;
    PyObject *out_obj = NULL;
    float *out;
    uint64_t bit_offset = 0;
    Py_ssize_t prob_offset;

    if (!PyArg_ParseTuple(args, "y*II", &payload, &expected_samples, &expected_alleles)) {
        return NULL;
    }
    data = (const unsigned char *)payload.buf;
    data_len = payload.len;

    if (data_len < 10) {
        PyBuffer_Release(&payload);
        PyErr_SetString(PyExc_ValueError, "Malformed BGEN genotype block: layout-2 header is truncated.");
        return NULL;
    }
    n_samples = read_u32_le(data);
    n_alleles = read_u16_le(data + 4);
    if (n_samples != expected_samples || n_alleles != expected_alleles) {
        PyBuffer_Release(&payload);
        PyErr_SetString(PyExc_ValueError, "Malformed BGEN genotype block: sample or allele count mismatch.");
        return NULL;
    }
    min_ploidy = data[6];
    max_ploidy = data[7];
    if (8 + (Py_ssize_t)n_samples + 2 > data_len) {
        PyBuffer_Release(&payload);
        PyErr_SetString(PyExc_ValueError, "Malformed BGEN genotype block: ploidy bytes are truncated.");
        return NULL;
    }
    ploidy_bytes = data + 8;
    phased = (int)data[8 + n_samples];
    bit_depth = data[8 + n_samples + 1];
    if (bit_depth < 1 || bit_depth > 32) {
        PyBuffer_Release(&payload);
        PyErr_SetString(PyExc_ValueError, "BGEN probability bit depth must be between 1 and 32.");
        return NULL;
    }
    if (max_probabilities(max_ploidy, n_alleles, phased, &width) < 0) {
        PyBuffer_Release(&payload);
        return NULL;
    }
    if (width == 0) {
        PyBuffer_Release(&payload);
        PyErr_SetString(PyExc_ValueError, "BGEN probability width cannot be zero.");
        return NULL;
    }
    if ((uint64_t)n_samples * width > (uint64_t)PY_SSIZE_T_MAX / sizeof(float)) {
        PyBuffer_Release(&payload);
        PyErr_SetString(PyExc_MemoryError, "BGEN probability array is too large.");
        return NULL;
    }
    out_obj = PyByteArray_FromStringAndSize(NULL, (Py_ssize_t)n_samples * (Py_ssize_t)width * (Py_ssize_t)sizeof(float));
    if (out_obj == NULL) {
        PyBuffer_Release(&payload);
        return NULL;
    }
    out = (float *)PyByteArray_AS_STRING(out_obj);
    denominator = bit_depth == 32 ? 4294967295.0 : (double)(((uint64_t)1 << bit_depth) - 1U);
    prob_offset = 10 + (Py_ssize_t)n_samples;

    for (uint32_t sample = 0; sample < n_samples; sample++) {
        uint8_t ploidy_byte = ploidy_bytes[sample];
        uint32_t ploidy = (uint32_t)(ploidy_byte & 0x3FU);
        int missing = (ploidy_byte & 0x80U) != 0;
        float *row = out + ((uint64_t)sample * width);

        if (ploidy < min_ploidy || ploidy > max_ploidy) {
            Py_DECREF(out_obj);
            PyBuffer_Release(&payload);
            PyErr_SetString(PyExc_ValueError, "Malformed BGEN genotype block: sample ploidy is out of range.");
            return NULL;
        }

        fill_nan(row, width);
        if (!phased) {
            uint32_t sample_width;
            uint32_t stored;
            double remainder = 1.0;
            if (max_probabilities(ploidy, n_alleles, 0, &sample_width) < 0) {
                Py_DECREF(out_obj);
                PyBuffer_Release(&payload);
                return NULL;
            }
            stored = sample_width > 0 ? sample_width - 1 : 0;
            for (uint32_t j = 0; j < stored; j++) {
                int ok = 1;
                uint32_t raw = read_bits_lsb(data + prob_offset, data_len - prob_offset, bit_offset, bit_depth, &ok);
                if (!ok) {
                    Py_DECREF(out_obj);
                    PyBuffer_Release(&payload);
                    PyErr_SetString(PyExc_ValueError, "Malformed BGEN genotype block: probability bits are truncated.");
                    return NULL;
                }
                bit_offset += bit_depth;
                if (!missing) {
                    double prob = (double)raw / denominator;
                    row[j] = (float)prob;
                    remainder -= prob;
                }
            }
            if (!missing && sample_width > 0) {
                row[stored] = (float)remainder;
            }
        } else {
            for (uint32_t hap = 0; hap < ploidy; hap++) {
                double remainder = 1.0;
                for (uint32_t allele = 0; allele + 1 < n_alleles; allele++) {
                    int ok = 1;
                    uint32_t raw = read_bits_lsb(data + prob_offset, data_len - prob_offset, bit_offset, bit_depth, &ok);
                    if (!ok) {
                        Py_DECREF(out_obj);
                        PyBuffer_Release(&payload);
                        PyErr_SetString(PyExc_ValueError, "Malformed BGEN genotype block: probability bits are truncated.");
                        return NULL;
                    }
                    bit_offset += bit_depth;
                    if (!missing) {
                        double prob = (double)raw / denominator;
                        row[hap * n_alleles + allele] = (float)prob;
                        remainder -= prob;
                    }
                }
                if (!missing) {
                    row[hap * n_alleles + (n_alleles - 1)] = (float)remainder;
                }
            }
        }
    }

    PyBuffer_Release(&payload);
    return Py_BuildValue("NIIiI", out_obj, n_samples, width, phased, bit_depth);
}

static double
read_probability(const Py_buffer *view, uint64_t index)
{
    const unsigned char *ptr = (const unsigned char *)view->buf + index * (uint64_t)view->itemsize;
    if (view->itemsize == 4) {
        float value;
        memcpy(&value, ptr, sizeof(float));
        return (double)value;
    }
    double value;
    memcpy(&value, ptr, sizeof(double));
    return value;
}

static int
is_sample_missing(const Py_buffer *view, uint64_t base, uint32_t count)
{
    int any_nan = 0;
    int any_finite = 0;
    for (uint32_t i = 0; i < count; i++) {
        double value = read_probability(view, base + i);
        if (isnan(value)) {
            any_nan = 1;
        } else {
            any_finite = 1;
        }
    }
    if (any_nan && any_finite) {
        PyErr_SetString(PyExc_ValueError, "BGEN missing genotypes must be encoded as all-NaN rows.");
        return -1;
    }
    return any_nan && !any_finite;
}

static uint32_t
scale_probability(double value, double factor)
{
    double scaled;
    if (value < 0.0 || value > 1.0) {
        PyErr_SetString(PyExc_ValueError, "BGEN probabilities must be in the [0, 1] range.");
        return 0;
    }
    scaled = value * factor + 0.5;
    if (scaled < 0.0) {
        scaled = 0.0;
    }
    if (scaled > factor) {
        scaled = factor;
    }
    return (uint32_t)scaled;
}

static PyObject *
encode_layout2(PyObject *self, PyObject *args)
{
    PyObject *prob_obj;
    PyObject *ploidy_obj;
    Py_buffer probs;
    Py_buffer ploidy_view;
    int have_ploidy_view = 0;
    unsigned int n_samples;
    unsigned int width;
    unsigned int n_alleles;
    unsigned int min_ploidy;
    unsigned int max_ploidy;
    int phased;
    unsigned int bit_depth;
    uint64_t stored_values = 0;
    uint64_t total_bits;
    Py_ssize_t payload_len;
    PyObject *payload_obj = NULL;
    unsigned char *payload;
    unsigned char *ploidy_bytes;
    double factor;
    uint64_t bit_offset = 0;

    if (!PyArg_ParseTuple(
            args,
            "OIIIIIiIO",
            &prob_obj,
            &n_samples,
            &width,
            &n_alleles,
            &min_ploidy,
            &max_ploidy,
            &phased,
            &bit_depth,
            &ploidy_obj)) {
        return NULL;
    }
    if (bit_depth < 1 || bit_depth > 32) {
        PyErr_SetString(PyExc_ValueError, "BGEN probability bit depth must be between 1 and 32.");
        return NULL;
    }
    if (n_samples == 0 || n_alleles == 0 || width == 0) {
        PyErr_SetString(PyExc_ValueError, "BGEN sample count, allele count, and probability width must be positive.");
        return NULL;
    }
    if (PyObject_GetBuffer(prob_obj, &probs, PyBUF_CONTIG_RO) < 0) {
        return NULL;
    }
    if (probs.itemsize != 4 && probs.itemsize != 8) {
        PyBuffer_Release(&probs);
        PyErr_SetString(PyExc_TypeError, "BGEN probability buffer must contain float32 or float64 values.");
        return NULL;
    }
    if ((uint64_t)probs.len < (uint64_t)n_samples * width * (uint64_t)probs.itemsize) {
        PyBuffer_Release(&probs);
        PyErr_SetString(PyExc_ValueError, "BGEN probability buffer is smaller than the declared shape.");
        return NULL;
    }
    if (ploidy_obj != Py_None) {
        if (PyObject_GetBuffer(ploidy_obj, &ploidy_view, PyBUF_CONTIG_RO) < 0) {
            PyBuffer_Release(&probs);
            return NULL;
        }
        have_ploidy_view = 1;
        if ((uint64_t)ploidy_view.len < n_samples) {
            PyBuffer_Release(&ploidy_view);
            PyBuffer_Release(&probs);
            PyErr_SetString(PyExc_ValueError, "BGEN ploidy buffer is smaller than the sample count.");
            return NULL;
        }
    }

    for (uint32_t sample = 0; sample < n_samples; sample++) {
        uint32_t ploidy = have_ploidy_view
            ? (uint32_t)((const unsigned char *)ploidy_view.buf)[sample]
            : (uint32_t)max_ploidy;
        uint32_t sample_width;
        if (ploidy < min_ploidy || ploidy > max_ploidy) {
            if (have_ploidy_view) {
                PyBuffer_Release(&ploidy_view);
            }
            PyBuffer_Release(&probs);
            PyErr_SetString(PyExc_ValueError, "BGEN ploidy value is out of range.");
            return NULL;
        }
        if (phased) {
            if ((uint64_t)ploidy * n_alleles > width) {
                if (have_ploidy_view) {
                    PyBuffer_Release(&ploidy_view);
                }
                PyBuffer_Release(&probs);
                PyErr_SetString(PyExc_ValueError, "BGEN phased probability width is incompatible with ploidy.");
                return NULL;
            }
            stored_values += (uint64_t)ploidy * (uint64_t)(n_alleles - 1);
        } else {
            if (max_probabilities(ploidy, n_alleles, 0, &sample_width) < 0) {
                if (have_ploidy_view) {
                    PyBuffer_Release(&ploidy_view);
                }
                PyBuffer_Release(&probs);
                return NULL;
            }
            if (sample_width > width) {
                if (have_ploidy_view) {
                    PyBuffer_Release(&ploidy_view);
                }
                PyBuffer_Release(&probs);
                PyErr_SetString(PyExc_ValueError, "BGEN unphased probability width is incompatible with ploidy.");
                return NULL;
            }
            stored_values += sample_width > 0 ? sample_width - 1 : 0;
        }
    }

    total_bits = stored_values * (uint64_t)bit_depth;
    if (total_bits > (uint64_t)(PY_SSIZE_T_MAX - 10 - (Py_ssize_t)n_samples) * 8U) {
        if (have_ploidy_view) {
            PyBuffer_Release(&ploidy_view);
        }
        PyBuffer_Release(&probs);
        PyErr_SetString(PyExc_MemoryError, "BGEN encoded probability block is too large.");
        return NULL;
    }
    payload_len = 10 + (Py_ssize_t)n_samples + (Py_ssize_t)((total_bits + 7U) / 8U);
    payload_obj = PyByteArray_FromStringAndSize(NULL, payload_len);
    if (payload_obj == NULL) {
        if (have_ploidy_view) {
            PyBuffer_Release(&ploidy_view);
        }
        PyBuffer_Release(&probs);
        return NULL;
    }
    payload = (unsigned char *)PyByteArray_AS_STRING(payload_obj);
    memset(payload, 0, payload_len);
    write_u32_le(payload, (uint32_t)n_samples);
    write_u16_le(payload + 4, (uint16_t)n_alleles);
    payload[6] = (unsigned char)min_ploidy;
    payload[7] = (unsigned char)max_ploidy;
    ploidy_bytes = payload + 8;
    payload[8 + n_samples] = phased ? 1 : 0;
    payload[8 + n_samples + 1] = (unsigned char)bit_depth;
    factor = bit_depth == 32 ? 4294967295.0 : (double)(((uint64_t)1 << bit_depth) - 1U);

    for (uint32_t sample = 0; sample < n_samples; sample++) {
        uint32_t ploidy = have_ploidy_view
            ? (uint32_t)((const unsigned char *)ploidy_view.buf)[sample]
            : (uint32_t)max_ploidy;
        uint64_t row_base = (uint64_t)sample * width;
        int missing;
        ploidy_bytes[sample] = (unsigned char)ploidy;

        if (phased) {
            uint32_t active = ploidy * n_alleles;
            missing = is_sample_missing(&probs, row_base, active);
            if (missing < 0) {
                Py_DECREF(payload_obj);
                if (have_ploidy_view) {
                    PyBuffer_Release(&ploidy_view);
                }
                PyBuffer_Release(&probs);
                return NULL;
            }
            if (missing) {
                ploidy_bytes[sample] |= 0x80U;
            }
            for (uint32_t hap = 0; hap < ploidy; hap++) {
                for (uint32_t allele = 0; allele + 1 < n_alleles; allele++) {
                    double value = missing ? 0.0 : read_probability(&probs, row_base + hap * n_alleles + allele);
                    uint32_t raw = scale_probability(value, factor);
                    if (PyErr_Occurred()) {
                        Py_DECREF(payload_obj);
                        if (have_ploidy_view) {
                            PyBuffer_Release(&ploidy_view);
                        }
                        PyBuffer_Release(&probs);
                        return NULL;
                    }
                    write_bits_lsb(payload + 10 + n_samples, bit_offset, bit_depth, raw);
                    bit_offset += bit_depth;
                }
            }
        } else {
            uint32_t sample_width;
            if (max_probabilities(ploidy, n_alleles, 0, &sample_width) < 0) {
                Py_DECREF(payload_obj);
                if (have_ploidy_view) {
                    PyBuffer_Release(&ploidy_view);
                }
                PyBuffer_Release(&probs);
                return NULL;
            }
            missing = is_sample_missing(&probs, row_base, sample_width);
            if (missing < 0) {
                Py_DECREF(payload_obj);
                if (have_ploidy_view) {
                    PyBuffer_Release(&ploidy_view);
                }
                PyBuffer_Release(&probs);
                return NULL;
            }
            if (missing) {
                ploidy_bytes[sample] |= 0x80U;
            }
            for (uint32_t j = 0; j + 1 < sample_width; j++) {
                double value = missing ? 0.0 : read_probability(&probs, row_base + j);
                uint32_t raw = scale_probability(value, factor);
                if (PyErr_Occurred()) {
                    Py_DECREF(payload_obj);
                    if (have_ploidy_view) {
                        PyBuffer_Release(&ploidy_view);
                    }
                    PyBuffer_Release(&probs);
                    return NULL;
                }
                write_bits_lsb(payload + 10 + n_samples, bit_offset, bit_depth, raw);
                bit_offset += bit_depth;
            }
        }
    }

    if (have_ploidy_view) {
        PyBuffer_Release(&ploidy_view);
    }
    PyBuffer_Release(&probs);
    return payload_obj;
}

static PyMethodDef BgenMethods[] = {
    {
        "read_file_probabilities",
        read_file_probabilities,
        METH_VARARGS,
        "Read all BGEN layout-2 genotype probabilities into one float32 NumPy array."
    },
    {
        "read_file_dosage",
        read_file_dosage,
        METH_VARARGS,
        "Read all biallelic BGEN layout-2 dosages into one float32 NumPy array."
    },
    {
        "decode_layout2",
        decode_layout2,
        METH_VARARGS,
        "Decode a BGEN layout-2 genotype probability block into float32 bytes."
    },
    {
        "encode_layout2",
        encode_layout2,
        METH_VARARGS,
        "Encode a float32/float64 probability row block as an uncompressed BGEN layout-2 genotype block."
    },
    {NULL, NULL, 0, NULL}
};

static struct PyModuleDef bgenmodule = {
    PyModuleDef_HEAD_INIT,
    "_bgen",
    NULL,
    -1,
    BgenMethods
};

PyMODINIT_FUNC
PyInit__bgen(void)
{
    return PyModule_Create(&bgenmodule);
}
