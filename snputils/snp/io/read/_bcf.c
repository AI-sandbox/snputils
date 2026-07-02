#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <stdint.h>
#include <limits.h>

static int
read_u32_le(const unsigned char *data, Py_ssize_t data_len, Py_ssize_t offset, uint32_t *value)
{
    if (offset < 0 || offset + 4 > data_len) {
        PyErr_SetString(PyExc_ValueError, "Malformed BCF: record header is truncated.");
        return -1;
    }
    *value = ((uint32_t)data[offset])
        | ((uint32_t)data[offset + 1] << 8)
        | ((uint32_t)data[offset + 2] << 16)
        | ((uint32_t)data[offset + 3] << 24);
    return 0;
}

static int
decode_gt_value(const unsigned char *ptr, Py_ssize_t type_size)
{
    uint32_t raw = 0;
    if (type_size == 1) {
        raw = ptr[0];
    } else if (type_size == 2) {
        raw = ((uint32_t)ptr[0]) | ((uint32_t)ptr[1] << 8);
    } else {
        raw = ((uint32_t)ptr[0])
            | ((uint32_t)ptr[1] << 8)
            | ((uint32_t)ptr[2] << 16)
            | ((uint32_t)ptr[3] << 24);
    }
    return (int)(raw >> 1) - 1;
}

static int
ensure_capacity(PyObject *buffer, Py_ssize_t *capacity_records, Py_ssize_t records_needed, Py_ssize_t row_width)
{
    Py_ssize_t new_capacity;
    Py_ssize_t new_size;

    if (records_needed <= *capacity_records) {
        return 0;
    }

    new_capacity = (*capacity_records > 0) ? *capacity_records : 1024;
    while (new_capacity < records_needed) {
        if (new_capacity > PY_SSIZE_T_MAX / 2) {
            PyErr_SetString(PyExc_MemoryError, "BCF genotype buffer is too large.");
            return -1;
        }
        new_capacity *= 2;
    }
    if (row_width != 0 && new_capacity > PY_SSIZE_T_MAX / row_width) {
        PyErr_SetString(PyExc_MemoryError, "BCF genotype buffer is too large.");
        return -1;
    }
    new_size = new_capacity * row_width;
    if (PyByteArray_Resize(buffer, new_size) < 0) {
        return -1;
    }
    *capacity_records = new_capacity;
    return 0;
}

static int
ensure_core_capacity(
    PyObject *gt_buffer,
    PyObject *chrom_buffer,
    PyObject *pos_buffer,
    PyObject *qual_buffer,
    PyObject *filter_buffer,
    Py_ssize_t *capacity_records,
    Py_ssize_t records_needed,
    Py_ssize_t gt_row_width)
{
    Py_ssize_t new_capacity;

    if (records_needed <= *capacity_records) {
        return 0;
    }

    new_capacity = (*capacity_records > 0) ? *capacity_records : 1024;
    while (new_capacity < records_needed) {
        if (new_capacity > PY_SSIZE_T_MAX / 2) {
            PyErr_SetString(PyExc_MemoryError, "BCF metadata buffer is too large.");
            return -1;
        }
        new_capacity *= 2;
    }

    if (gt_row_width != 0 && new_capacity > PY_SSIZE_T_MAX / gt_row_width) {
        PyErr_SetString(PyExc_MemoryError, "BCF genotype buffer is too large.");
        return -1;
    }
    if (PyByteArray_Resize(gt_buffer, new_capacity * gt_row_width) < 0
        || PyByteArray_Resize(chrom_buffer, new_capacity * 4) < 0
        || PyByteArray_Resize(pos_buffer, new_capacity * 8) < 0
        || PyByteArray_Resize(qual_buffer, new_capacity * 4) < 0
        || PyByteArray_Resize(filter_buffer, new_capacity) < 0) {
        return -1;
    }
    *capacity_records = new_capacity;
    return 0;
}

static int
type_size_from_code(int type_code)
{
    switch (type_code) {
        case 0:
            return 0;
        case 1:
            return 1;
        case 2:
            return 2;
        case 3:
            return 4;
        case 5:
            return 4;
        case 7:
            return 1;
        default:
            return -1;
    }
}

static int
read_typed_descriptor(
    const unsigned char *data,
    Py_ssize_t data_len,
    Py_ssize_t *offset,
    Py_ssize_t *n_vals,
    int *type_code,
    Py_ssize_t *type_size)
{
    unsigned char descriptor;
    Py_ssize_t n;

    if (*offset < 0 || *offset >= data_len) {
        PyErr_SetString(PyExc_ValueError, "Malformed BCF: typed descriptor is truncated.");
        return -1;
    }

    descriptor = data[*offset];
    (*offset)++;
    *type_code = descriptor & 0x0F;
    n = descriptor >> 4;

    *type_size = type_size_from_code(*type_code);
    if (*type_size < 0) {
        PyErr_SetString(PyExc_ValueError, "Unsupported BCF atomic type code.");
        return -1;
    }

    if (n == 15) {
        int length_type;
        Py_ssize_t length_size;
        Py_ssize_t value = 0;
        if (*offset >= data_len) {
            PyErr_SetString(PyExc_ValueError, "Malformed BCF: typed length descriptor is truncated.");
            return -1;
        }
        length_type = data[*offset] & 0x0F;
        (*offset)++;
        length_size = type_size_from_code(length_type);
        if ((length_type != 1 && length_type != 2 && length_type != 3) || length_size <= 0) {
            PyErr_SetString(PyExc_ValueError, "Cannot identify the BCF typed-value length encoding.");
            return -1;
        }
        if (*offset + length_size > data_len) {
            PyErr_SetString(PyExc_ValueError, "Malformed BCF: typed value length is truncated.");
            return -1;
        }
        for (Py_ssize_t i = 0; i < length_size; i++) {
            value |= ((Py_ssize_t)data[*offset + i]) << (8 * i);
        }
        *offset += length_size;
        n = value;
    }

    *n_vals = n;
    return 0;
}

static Py_ssize_t
trim_nul(const unsigned char *start, Py_ssize_t n_vals)
{
    for (Py_ssize_t i = 0; i < n_vals; i++) {
        if (start[i] == '\0') {
            return i;
        }
    }
    return n_vals;
}

static PyObject *
read_typed_string_object(
    const unsigned char *data,
    Py_ssize_t data_len,
    Py_ssize_t *offset,
    PyObject *cache,
    PyObject *dot,
    int empty_as_dot)
{
    Py_ssize_t n_vals, type_size, end, trimmed_len;
    int type_code;
    const unsigned char *start;

    if (read_typed_descriptor(data, data_len, offset, &n_vals, &type_code, &type_size) < 0) {
        return NULL;
    }
    if (type_code != 7 || type_size != 1) {
        PyErr_SetString(PyExc_ValueError, "Expected a typed string in the BCF record.");
        return NULL;
    }
    if (n_vals < 0 || *offset > data_len || n_vals > data_len - *offset) {
        PyErr_SetString(PyExc_ValueError, "Malformed BCF: typed string extends beyond end of file.");
        return NULL;
    }

    start = data + *offset;
    trimmed_len = trim_nul(start, n_vals);
    end = *offset + n_vals;
    *offset = end;

    if (empty_as_dot && (trimmed_len == 0 || (trimmed_len == 1 && start[0] == '.'))) {
        Py_INCREF(dot);
        return dot;
    }

    if (cache != NULL) {
        PyObject *key = PyBytes_FromStringAndSize((const char *)start, trimmed_len);
        PyObject *found;
        PyObject *value;
        if (key == NULL) {
            return NULL;
        }
        found = PyDict_GetItemWithError(cache, key);
        if (found != NULL) {
            Py_INCREF(found);
            Py_DECREF(key);
            return found;
        }
        if (PyErr_Occurred()) {
            Py_DECREF(key);
            return NULL;
        }
        value = PyUnicode_FromStringAndSize((const char *)start, trimmed_len);
        if (value == NULL) {
            Py_DECREF(key);
            return NULL;
        }
        if (PyDict_SetItem(cache, key, value) < 0) {
            Py_DECREF(value);
            Py_DECREF(key);
            return NULL;
        }
        Py_DECREF(key);
        return value;
    }

    return PyUnicode_FromStringAndSize((const char *)start, trimmed_len);
}

static int
read_i32_le_raw(const unsigned char *data, Py_ssize_t offset)
{
    uint32_t raw = ((uint32_t)data[offset])
        | ((uint32_t)data[offset + 1] << 8)
        | ((uint32_t)data[offset + 2] << 16)
        | ((uint32_t)data[offset + 3] << 24);
    return (int32_t)raw;
}

static void
write_i32_le(char *out, int32_t value)
{
    uint32_t raw = (uint32_t)value;
    out[0] = (char)(raw & 0xFF);
    out[1] = (char)((raw >> 8) & 0xFF);
    out[2] = (char)((raw >> 16) & 0xFF);
    out[3] = (char)((raw >> 24) & 0xFF);
}

static void
write_u32_le(char *out, uint32_t value)
{
    out[0] = (char)(value & 0xFF);
    out[1] = (char)((value >> 8) & 0xFF);
    out[2] = (char)((value >> 16) & 0xFF);
    out[3] = (char)((value >> 24) & 0xFF);
}

static void
write_i64_le(char *out, int64_t value)
{
    uint64_t raw = (uint64_t)value;
    for (int i = 0; i < 8; i++) {
        out[i] = (char)((raw >> (8 * i)) & 0xFF);
    }
}

static uint32_t
read_int_value_unsigned(const unsigned char *ptr, Py_ssize_t type_size)
{
    if (type_size == 1) {
        return ptr[0];
    }
    if (type_size == 2) {
        return ((uint32_t)ptr[0]) | ((uint32_t)ptr[1] << 8);
    }
    return ((uint32_t)ptr[0])
        | ((uint32_t)ptr[1] << 8)
        | ((uint32_t)ptr[2] << 16)
        | ((uint32_t)ptr[3] << 24);
}

static int
parse_filter_pass(
    const unsigned char *data,
    Py_ssize_t data_len,
    Py_ssize_t *offset,
    int pass_filter_id,
    unsigned char *out_pass)
{
    Py_ssize_t n_vals, type_size, values_offset;
    int type_code;
    uint32_t missing;
    uint32_t vector_end;
    int n_filters = 0;
    uint32_t last_filter = 0;

    if (read_typed_descriptor(data, data_len, offset, &n_vals, &type_code, &type_size) < 0) {
        return -1;
    }
    if (type_code == 0) {
        *out_pass = 1;
        return 0;
    }
    if (type_code != 1 && type_code != 2 && type_code != 3) {
        PyErr_SetString(PyExc_ValueError, "Expected an integer FILTER typed value in the BCF record.");
        return -1;
    }
    if (n_vals < 0 || *offset > data_len || n_vals > (data_len - *offset) / type_size) {
        PyErr_SetString(PyExc_ValueError, "Malformed BCF: FILTER field extends beyond end of file.");
        return -1;
    }

    values_offset = *offset;
    missing = (uint32_t)1 << ((int)type_size * 8 - 1);
    vector_end = missing | 1u;
    for (Py_ssize_t i = 0; i < n_vals; i++) {
        uint32_t raw = read_int_value_unsigned(data + values_offset + i * type_size, type_size);
        if (raw == vector_end) {
            break;
        }
        if (raw == missing) {
            continue;
        }
        n_filters++;
        last_filter = raw;
    }
    *offset += n_vals * type_size;
    *out_pass = (unsigned char)(n_filters == 0 || (n_filters == 1 && (int)last_filter == pass_filter_id));
    return 0;
}

static PyObject *
decode_gt(PyObject *self, PyObject *args)
{
    PyObject *data_obj = NULL;
    PyObject *sample_indices_obj = NULL;
    PyObject *sample_seq = NULL;
    Py_buffer data_view;
    PyObject *out = NULL;
    Py_ssize_t *sample_indices = NULL;
    Py_ssize_t body_offset, gt_rel_offset, n_samples, n_vals, type_size, expected_l_indiv;
    Py_ssize_t n_selected, row_width, capacity_records, output_size;
    Py_ssize_t offset, n_records;
    int sum_strands;
    int all_samples;
    const unsigned char *data;
    Py_ssize_t data_len;

    data_view.buf = NULL;
    if (!PyArg_ParseTuple(
            args,
            "OnnnnnnOp",
            &data_obj,
            &body_offset,
            &gt_rel_offset,
            &n_samples,
            &n_vals,
            &type_size,
            &expected_l_indiv,
            &sample_indices_obj,
            &sum_strands)) {
        return NULL;
    }

    if (n_samples < 0 || n_vals < 1 || n_vals > 2 || (type_size != 1 && type_size != 2 && type_size != 4)) {
        PyErr_SetString(PyExc_ValueError, "Unsupported BCF FORMAT/GT layout.");
        return NULL;
    }

    all_samples = (sample_indices_obj == Py_None);
    if (all_samples) {
        n_selected = n_samples;
    } else {
        sample_seq = PySequence_Fast(sample_indices_obj, "sample indices must be a sequence");
        if (sample_seq == NULL) {
            return NULL;
        }
        n_selected = PySequence_Fast_GET_SIZE(sample_seq);
        sample_indices = PyMem_New(Py_ssize_t, n_selected);
        if (sample_indices == NULL) {
            Py_DECREF(sample_seq);
            PyErr_NoMemory();
            return NULL;
        }
        for (Py_ssize_t i = 0; i < n_selected; i++) {
            PyObject *item = PySequence_Fast_GET_ITEM(sample_seq, i);
            Py_ssize_t idx = PyLong_AsSsize_t(item);
            if (idx == -1 && PyErr_Occurred()) {
                PyMem_Free(sample_indices);
                Py_DECREF(sample_seq);
                return NULL;
            }
            if (idx < 0 || idx >= n_samples) {
                PyMem_Free(sample_indices);
                Py_DECREF(sample_seq);
                PyErr_SetString(PyExc_ValueError, "One or more sample indexes are out of bounds.");
                return NULL;
            }
            sample_indices[i] = idx;
        }
        Py_DECREF(sample_seq);
        sample_seq = NULL;
    }

    if (!sum_strands && n_selected > PY_SSIZE_T_MAX / 2) {
        PyMem_Free(sample_indices);
        PyErr_SetString(PyExc_MemoryError, "BCF genotype row is too large.");
        return NULL;
    }
    row_width = sum_strands ? n_selected : n_selected * 2;

    if (PyObject_GetBuffer(data_obj, &data_view, PyBUF_SIMPLE) < 0) {
        PyMem_Free(sample_indices);
        return NULL;
    }
    data = (const unsigned char *)data_view.buf;
    data_len = data_view.len;
    if (body_offset < 0 || body_offset > data_len) {
        PyBuffer_Release(&data_view);
        PyMem_Free(sample_indices);
        PyErr_SetString(PyExc_ValueError, "BCF body offset is out of bounds.");
        return NULL;
    }

    if (body_offset < data_len) {
        uint32_t first_l_shared, first_l_indiv;
        Py_ssize_t first_total;
        if (read_u32_le(data, data_len, body_offset, &first_l_shared) < 0
            || read_u32_le(data, data_len, body_offset + 4, &first_l_indiv) < 0) {
            PyBuffer_Release(&data_view);
            PyMem_Free(sample_indices);
            return NULL;
        }
        first_total = 8 + (Py_ssize_t)first_l_shared + (Py_ssize_t)first_l_indiv;
        capacity_records = first_total > 0 ? ((data_len - body_offset) / first_total) + 1024 : 1024;
    } else {
        capacity_records = 0;
    }

    if (row_width != 0 && capacity_records > PY_SSIZE_T_MAX / row_width) {
        PyBuffer_Release(&data_view);
        PyMem_Free(sample_indices);
        PyErr_SetString(PyExc_MemoryError, "BCF genotype buffer is too large.");
        return NULL;
    }
    out = PyByteArray_FromStringAndSize(NULL, capacity_records * row_width);
    if (out == NULL) {
        PyBuffer_Release(&data_view);
        PyMem_Free(sample_indices);
        return NULL;
    }

    offset = body_offset;
    n_records = 0;
    while (offset < data_len) {
        uint32_t l_shared_u32, l_indiv_u32;
        Py_ssize_t l_shared, l_indiv, indiv_offset, gt_offset, record_end, gt_span;
        char *row;

        if (read_u32_le(data, data_len, offset, &l_shared_u32) < 0
            || read_u32_le(data, data_len, offset + 4, &l_indiv_u32) < 0) {
            Py_DECREF(out);
            PyBuffer_Release(&data_view);
            PyMem_Free(sample_indices);
            return NULL;
        }
        l_shared = (Py_ssize_t)l_shared_u32;
        l_indiv = (Py_ssize_t)l_indiv_u32;
        if (expected_l_indiv >= 0 && l_indiv != expected_l_indiv) {
            Py_DECREF(out);
            PyBuffer_Release(&data_view);
            PyMem_Free(sample_indices);
            Py_RETURN_NONE;
        }

        indiv_offset = offset + 8 + l_shared;
        record_end = indiv_offset + l_indiv;
        gt_offset = indiv_offset + gt_rel_offset;
        if (indiv_offset < offset || record_end < indiv_offset || record_end > data_len || gt_offset < indiv_offset) {
            Py_DECREF(out);
            PyBuffer_Release(&data_view);
            PyMem_Free(sample_indices);
            PyErr_SetString(PyExc_ValueError, "Malformed BCF: record extends beyond end of file.");
            return NULL;
        }

        if (n_samples != 0 && n_vals > PY_SSIZE_T_MAX / n_samples) {
            Py_DECREF(out);
            PyBuffer_Release(&data_view);
            PyMem_Free(sample_indices);
            PyErr_SetString(PyExc_MemoryError, "BCF FORMAT/GT field is too large.");
            return NULL;
        }
        gt_span = n_samples * n_vals;
        if (gt_span != 0 && type_size > PY_SSIZE_T_MAX / gt_span) {
            Py_DECREF(out);
            PyBuffer_Release(&data_view);
            PyMem_Free(sample_indices);
            PyErr_SetString(PyExc_MemoryError, "BCF FORMAT/GT field is too large.");
            return NULL;
        }
        gt_span *= type_size;
        if (gt_offset + gt_span > record_end) {
            Py_DECREF(out);
            PyBuffer_Release(&data_view);
            PyMem_Free(sample_indices);
            Py_RETURN_NONE;
        }

        if (ensure_capacity(out, &capacity_records, n_records + 1, row_width) < 0) {
            Py_DECREF(out);
            PyBuffer_Release(&data_view);
            PyMem_Free(sample_indices);
            return NULL;
        }
        row = PyByteArray_AS_STRING(out) + n_records * row_width;

        if (sum_strands) {
            for (Py_ssize_t out_sample = 0; out_sample < n_selected; out_sample++) {
                Py_ssize_t sample = all_samples ? out_sample : sample_indices[out_sample];
                const unsigned char *gt = data + gt_offset + sample * n_vals * type_size;
                int first = decode_gt_value(gt, type_size);
                int value = (n_vals == 1) ? (first - 1) : (first + decode_gt_value(gt + type_size, type_size));
                row[out_sample] = (char)value;
            }
        } else {
            for (Py_ssize_t out_sample = 0; out_sample < n_selected; out_sample++) {
                Py_ssize_t sample = all_samples ? out_sample : sample_indices[out_sample];
                const unsigned char *gt = data + gt_offset + sample * n_vals * type_size;
                row[out_sample * 2] = (char)decode_gt_value(gt, type_size);
                row[out_sample * 2 + 1] = (n_vals == 1) ? (char)-1 : (char)decode_gt_value(gt + type_size, type_size);
            }
        }

        offset = record_end;
        n_records++;
    }

    if (offset != data_len) {
        Py_DECREF(out);
        PyBuffer_Release(&data_view);
        PyMem_Free(sample_indices);
        PyErr_SetString(PyExc_ValueError, "Malformed BCF: record boundaries do not consume the full file.");
        return NULL;
    }

    output_size = n_records * row_width;
    if (PyByteArray_Resize(out, output_size) < 0) {
        Py_DECREF(out);
        PyBuffer_Release(&data_view);
        PyMem_Free(sample_indices);
        return NULL;
    }

    PyBuffer_Release(&data_view);
    PyMem_Free(sample_indices);
    return Py_BuildValue("Nn", out, n_records);
}

static PyObject *
decode_core(PyObject *self, PyObject *args)
{
    PyObject *data_obj = NULL;
    PyObject *sample_indices_obj = NULL;
    PyObject *sample_seq = NULL;
    Py_buffer data_view;
    PyObject *gt_out = NULL;
    PyObject *chrom_out = NULL;
    PyObject *pos_out = NULL;
    PyObject *qual_out = NULL;
    PyObject *filter_out = NULL;
    PyObject *ids = NULL;
    PyObject *refs = NULL;
    PyObject *alts = NULL;
    PyObject *ref_cache = NULL;
    PyObject *alt_cache = NULL;
    PyObject *dot = NULL;
    Py_ssize_t *sample_indices = NULL;
    Py_ssize_t body_offset, gt_rel_offset, n_samples, n_vals, type_size, expected_l_indiv;
    Py_ssize_t n_selected, gt_row_width, capacity_records, output_size;
    Py_ssize_t offset, n_records;
    int sum_strands;
    int all_samples;
    int pass_filter_id;
    const unsigned char *data;
    Py_ssize_t data_len;

    data_view.buf = NULL;
    if (!PyArg_ParseTuple(
            args,
            "OnnnnnnOpi",
            &data_obj,
            &body_offset,
            &gt_rel_offset,
            &n_samples,
            &n_vals,
            &type_size,
            &expected_l_indiv,
            &sample_indices_obj,
            &sum_strands,
            &pass_filter_id)) {
        return NULL;
    }

    if (n_samples < 0 || n_vals < 1 || n_vals > 2 || (type_size != 1 && type_size != 2 && type_size != 4)) {
        PyErr_SetString(PyExc_ValueError, "Unsupported BCF FORMAT/GT layout.");
        return NULL;
    }

    all_samples = (sample_indices_obj == Py_None);
    if (all_samples) {
        n_selected = n_samples;
    } else {
        sample_seq = PySequence_Fast(sample_indices_obj, "sample indices must be a sequence");
        if (sample_seq == NULL) {
            return NULL;
        }
        n_selected = PySequence_Fast_GET_SIZE(sample_seq);
        sample_indices = PyMem_New(Py_ssize_t, n_selected);
        if (sample_indices == NULL) {
            Py_DECREF(sample_seq);
            PyErr_NoMemory();
            return NULL;
        }
        for (Py_ssize_t i = 0; i < n_selected; i++) {
            PyObject *item = PySequence_Fast_GET_ITEM(sample_seq, i);
            Py_ssize_t idx = PyLong_AsSsize_t(item);
            if (idx == -1 && PyErr_Occurred()) {
                PyMem_Free(sample_indices);
                Py_DECREF(sample_seq);
                return NULL;
            }
            if (idx < 0 || idx >= n_samples) {
                PyMem_Free(sample_indices);
                Py_DECREF(sample_seq);
                PyErr_SetString(PyExc_ValueError, "One or more sample indexes are out of bounds.");
                return NULL;
            }
            sample_indices[i] = idx;
        }
        Py_DECREF(sample_seq);
        sample_seq = NULL;
    }

    if (!sum_strands && n_selected > PY_SSIZE_T_MAX / 2) {
        PyMem_Free(sample_indices);
        PyErr_SetString(PyExc_MemoryError, "BCF genotype row is too large.");
        return NULL;
    }
    gt_row_width = sum_strands ? n_selected : n_selected * 2;

    if (PyObject_GetBuffer(data_obj, &data_view, PyBUF_SIMPLE) < 0) {
        PyMem_Free(sample_indices);
        return NULL;
    }
    data = (const unsigned char *)data_view.buf;
    data_len = data_view.len;
    if (body_offset < 0 || body_offset > data_len) {
        PyBuffer_Release(&data_view);
        PyMem_Free(sample_indices);
        PyErr_SetString(PyExc_ValueError, "BCF body offset is out of bounds.");
        return NULL;
    }

    if (body_offset < data_len) {
        uint32_t first_l_shared, first_l_indiv;
        Py_ssize_t first_total;
        if (read_u32_le(data, data_len, body_offset, &first_l_shared) < 0
            || read_u32_le(data, data_len, body_offset + 4, &first_l_indiv) < 0) {
            PyBuffer_Release(&data_view);
            PyMem_Free(sample_indices);
            return NULL;
        }
        first_total = 8 + (Py_ssize_t)first_l_shared + (Py_ssize_t)first_l_indiv;
        capacity_records = first_total > 0 ? ((data_len - body_offset) / first_total) + 1024 : 1024;
    } else {
        capacity_records = 0;
    }

    if ((gt_row_width != 0 && capacity_records > PY_SSIZE_T_MAX / gt_row_width)
        || capacity_records > PY_SSIZE_T_MAX / 8) {
        PyBuffer_Release(&data_view);
        PyMem_Free(sample_indices);
        PyErr_SetString(PyExc_MemoryError, "BCF core metadata buffers are too large.");
        return NULL;
    }

    gt_out = PyByteArray_FromStringAndSize(NULL, capacity_records * gt_row_width);
    chrom_out = PyByteArray_FromStringAndSize(NULL, capacity_records * 4);
    pos_out = PyByteArray_FromStringAndSize(NULL, capacity_records * 8);
    qual_out = PyByteArray_FromStringAndSize(NULL, capacity_records * 4);
    filter_out = PyByteArray_FromStringAndSize(NULL, capacity_records);
    ids = PyList_New(0);
    refs = PyList_New(0);
    alts = PyList_New(0);
    ref_cache = PyDict_New();
    alt_cache = PyDict_New();
    dot = PyUnicode_FromString(".");
    if (gt_out == NULL || chrom_out == NULL || pos_out == NULL || qual_out == NULL
        || filter_out == NULL || ids == NULL || refs == NULL || alts == NULL
        || ref_cache == NULL || alt_cache == NULL || dot == NULL) {
        goto error;
    }

    offset = body_offset;
    n_records = 0;
    while (offset < data_len) {
        uint32_t l_shared_u32, l_indiv_u32, n_alleles_info_u32, n_fmt_samples_u32, qual_raw;
        Py_ssize_t l_shared, l_indiv, base, indiv_offset, gt_offset, record_end, gt_span;
        Py_ssize_t shared_offset;
        int32_t chrom_id, pos0;
        uint32_t n_alleles;
        unsigned char filter_pass;
        PyObject *variant_id = NULL;
        PyObject *ref = NULL;
        PyObject *alt = NULL;
        char *gt_row;

        if (read_u32_le(data, data_len, offset, &l_shared_u32) < 0
            || read_u32_le(data, data_len, offset + 4, &l_indiv_u32) < 0) {
            goto error;
        }
        l_shared = (Py_ssize_t)l_shared_u32;
        l_indiv = (Py_ssize_t)l_indiv_u32;
        if (expected_l_indiv >= 0 && l_indiv != expected_l_indiv) {
            Py_DECREF(gt_out);
            Py_DECREF(chrom_out);
            Py_DECREF(pos_out);
            Py_DECREF(qual_out);
            Py_DECREF(filter_out);
            Py_DECREF(ids);
            Py_DECREF(refs);
            Py_DECREF(alts);
            Py_DECREF(ref_cache);
            Py_DECREF(alt_cache);
            Py_DECREF(dot);
            PyBuffer_Release(&data_view);
            PyMem_Free(sample_indices);
            Py_RETURN_NONE;
        }

        base = offset + 8;
        indiv_offset = base + l_shared;
        record_end = indiv_offset + l_indiv;
        gt_offset = indiv_offset + gt_rel_offset;
        if (base < offset || base + 24 > data_len || indiv_offset < base || record_end < indiv_offset || record_end > data_len) {
            PyErr_SetString(PyExc_ValueError, "Malformed BCF: record extends beyond end of file.");
            goto error;
        }

        if (read_u32_le(data, data_len, base + 12, &qual_raw) < 0
            || read_u32_le(data, data_len, base + 16, &n_alleles_info_u32) < 0
            || read_u32_le(data, data_len, base + 20, &n_fmt_samples_u32) < 0) {
            goto error;
        }
        chrom_id = read_i32_le_raw(data, base);
        pos0 = read_i32_le_raw(data, base + 4);
        n_alleles = n_alleles_info_u32 >> 16;
        if ((n_fmt_samples_u32 & 0xFFFFFFu) != (uint32_t)n_samples) {
            PyErr_SetString(PyExc_ValueError, "BCF record sample count does not match header sample count.");
            goto error;
        }

        if (n_samples != 0 && n_vals > PY_SSIZE_T_MAX / n_samples) {
            PyErr_SetString(PyExc_MemoryError, "BCF FORMAT/GT field is too large.");
            goto error;
        }
        gt_span = n_samples * n_vals;
        if (gt_span != 0 && type_size > PY_SSIZE_T_MAX / gt_span) {
            PyErr_SetString(PyExc_MemoryError, "BCF FORMAT/GT field is too large.");
            goto error;
        }
        gt_span *= type_size;
        if (gt_offset < indiv_offset || gt_offset + gt_span > record_end) {
            Py_DECREF(gt_out);
            Py_DECREF(chrom_out);
            Py_DECREF(pos_out);
            Py_DECREF(qual_out);
            Py_DECREF(filter_out);
            Py_DECREF(ids);
            Py_DECREF(refs);
            Py_DECREF(alts);
            Py_DECREF(ref_cache);
            Py_DECREF(alt_cache);
            Py_DECREF(dot);
            PyBuffer_Release(&data_view);
            PyMem_Free(sample_indices);
            Py_RETURN_NONE;
        }

        shared_offset = base + 24;
        if (shared_offset > indiv_offset) {
            PyErr_SetString(PyExc_ValueError, "Malformed BCF: shared section is truncated.");
            goto error;
        }

        variant_id = read_typed_string_object(data, data_len, &shared_offset, NULL, dot, 1);
        ref = read_typed_string_object(data, data_len, &shared_offset, ref_cache, dot, 0);
        if (variant_id == NULL || ref == NULL) {
            Py_XDECREF(variant_id);
            Py_XDECREF(ref);
            goto error;
        }

        if (n_alleles <= 1) {
            alt = PyUnicode_FromString("");
            if (alt == NULL) {
                Py_DECREF(variant_id);
                Py_DECREF(ref);
                goto error;
            }
        } else if (n_alleles == 2) {
            alt = read_typed_string_object(data, data_len, &shared_offset, alt_cache, dot, 0);
            if (alt == NULL) {
                Py_DECREF(variant_id);
                Py_DECREF(ref);
                goto error;
            }
        } else {
            PyObject *alt_items = PyList_New(0);
            PyObject *comma = PyUnicode_FromString(",");
            if (alt_items == NULL || comma == NULL) {
                Py_XDECREF(alt_items);
                Py_XDECREF(comma);
                Py_DECREF(variant_id);
                Py_DECREF(ref);
                goto error;
            }
            for (uint32_t allele_i = 1; allele_i < n_alleles; allele_i++) {
                PyObject *one_alt = read_typed_string_object(data, data_len, &shared_offset, alt_cache, dot, 0);
                if (one_alt == NULL || PyList_Append(alt_items, one_alt) < 0) {
                    Py_XDECREF(one_alt);
                    Py_DECREF(alt_items);
                    Py_DECREF(comma);
                    Py_DECREF(variant_id);
                    Py_DECREF(ref);
                    goto error;
                }
                Py_DECREF(one_alt);
            }
            alt = PyUnicode_Join(comma, alt_items);
            Py_DECREF(alt_items);
            Py_DECREF(comma);
            if (alt == NULL) {
                Py_DECREF(variant_id);
                Py_DECREF(ref);
                goto error;
            }
        }

        if (shared_offset > indiv_offset || parse_filter_pass(data, data_len, &shared_offset, pass_filter_id, &filter_pass) < 0) {
            Py_DECREF(variant_id);
            Py_DECREF(ref);
            Py_DECREF(alt);
            goto error;
        }

        if (ensure_core_capacity(
                gt_out, chrom_out, pos_out, qual_out, filter_out,
                &capacity_records, n_records + 1, gt_row_width) < 0) {
            Py_DECREF(variant_id);
            Py_DECREF(ref);
            Py_DECREF(alt);
            goto error;
        }

        gt_row = PyByteArray_AS_STRING(gt_out) + n_records * gt_row_width;
        if (sum_strands) {
            for (Py_ssize_t out_sample = 0; out_sample < n_selected; out_sample++) {
                Py_ssize_t sample = all_samples ? out_sample : sample_indices[out_sample];
                const unsigned char *gt = data + gt_offset + sample * n_vals * type_size;
                int first = decode_gt_value(gt, type_size);
                int value = (n_vals == 1) ? (first - 1) : (first + decode_gt_value(gt + type_size, type_size));
                gt_row[out_sample] = (char)value;
            }
        } else {
            for (Py_ssize_t out_sample = 0; out_sample < n_selected; out_sample++) {
                Py_ssize_t sample = all_samples ? out_sample : sample_indices[out_sample];
                const unsigned char *gt = data + gt_offset + sample * n_vals * type_size;
                gt_row[out_sample * 2] = (char)decode_gt_value(gt, type_size);
                gt_row[out_sample * 2 + 1] = (n_vals == 1) ? (char)-1 : (char)decode_gt_value(gt + type_size, type_size);
            }
        }

        write_i32_le(PyByteArray_AS_STRING(chrom_out) + n_records * 4, chrom_id);
        write_i64_le(PyByteArray_AS_STRING(pos_out) + n_records * 8, (int64_t)pos0 + 1);
        write_u32_le(PyByteArray_AS_STRING(qual_out) + n_records * 4, qual_raw);
        PyByteArray_AS_STRING(filter_out)[n_records] = (char)filter_pass;

        if (PyList_Append(ids, variant_id) < 0 || PyList_Append(refs, ref) < 0 || PyList_Append(alts, alt) < 0) {
            Py_DECREF(variant_id);
            Py_DECREF(ref);
            Py_DECREF(alt);
            goto error;
        }
        Py_DECREF(variant_id);
        Py_DECREF(ref);
        Py_DECREF(alt);

        offset = record_end;
        n_records++;
    }

    if (offset != data_len) {
        PyErr_SetString(PyExc_ValueError, "Malformed BCF: record boundaries do not consume the full file.");
        goto error;
    }

    output_size = n_records * gt_row_width;
    if (PyByteArray_Resize(gt_out, output_size) < 0
        || PyByteArray_Resize(chrom_out, n_records * 4) < 0
        || PyByteArray_Resize(pos_out, n_records * 8) < 0
        || PyByteArray_Resize(qual_out, n_records * 4) < 0
        || PyByteArray_Resize(filter_out, n_records) < 0) {
        goto error;
    }

    Py_DECREF(ref_cache);
    Py_DECREF(alt_cache);
    Py_DECREF(dot);
    PyBuffer_Release(&data_view);
    PyMem_Free(sample_indices);

    return Py_BuildValue(
        "NNNNNNNNn",
        gt_out,
        chrom_out,
        pos_out,
        qual_out,
        filter_out,
        ids,
        refs,
        alts,
        n_records);

error:
    Py_XDECREF(gt_out);
    Py_XDECREF(chrom_out);
    Py_XDECREF(pos_out);
    Py_XDECREF(qual_out);
    Py_XDECREF(filter_out);
    Py_XDECREF(ids);
    Py_XDECREF(refs);
    Py_XDECREF(alts);
    Py_XDECREF(ref_cache);
    Py_XDECREF(alt_cache);
    Py_XDECREF(dot);
    if (data_view.buf != NULL) {
        PyBuffer_Release(&data_view);
    }
    PyMem_Free(sample_indices);
    return NULL;
}

static PyObject *
select_region_offsets(PyObject *self, PyObject *args)
{
    PyObject *data_obj = NULL;
    Py_buffer data_view;
    PyObject *offsets_out = NULL;
    const unsigned char *data;
    Py_ssize_t data_len;
    Py_ssize_t body_offset;
    int contig_id;
    long long start_pos;
    long long end_pos;
    Py_ssize_t offset;
    Py_ssize_t n_records = 0;
    Py_ssize_t n_selected = 0;
    Py_ssize_t capacity = 1024;

    data_view.buf = NULL;
    if (!PyArg_ParseTuple(args, "OniLL", &data_obj, &body_offset, &contig_id, &start_pos, &end_pos)) {
        return NULL;
    }
    if (PyObject_GetBuffer(data_obj, &data_view, PyBUF_SIMPLE) < 0) {
        return NULL;
    }
    data = (const unsigned char *)data_view.buf;
    data_len = data_view.len;
    if (body_offset < 0 || body_offset > data_len) {
        PyBuffer_Release(&data_view);
        PyErr_SetString(PyExc_ValueError, "BCF body offset is out of bounds.");
        return NULL;
    }

    offsets_out = PyByteArray_FromStringAndSize(NULL, capacity * 8);
    if (offsets_out == NULL) {
        PyBuffer_Release(&data_view);
        return NULL;
    }

    offset = body_offset;
    while (offset < data_len) {
        uint32_t l_shared_u32, l_indiv_u32;
        Py_ssize_t l_shared, l_indiv, base, record_end;
        int32_t record_contig;
        int32_t pos0;
        long long pos1;

        if (read_u32_le(data, data_len, offset, &l_shared_u32) < 0
            || read_u32_le(data, data_len, offset + 4, &l_indiv_u32) < 0) {
            Py_DECREF(offsets_out);
            PyBuffer_Release(&data_view);
            return NULL;
        }
        l_shared = (Py_ssize_t)l_shared_u32;
        l_indiv = (Py_ssize_t)l_indiv_u32;
        base = offset + 8;
        record_end = base + l_shared + l_indiv;
        if (base < offset || base + 8 > data_len || record_end < base || record_end > data_len) {
            Py_DECREF(offsets_out);
            PyBuffer_Release(&data_view);
            PyErr_SetString(PyExc_ValueError, "Malformed BCF: record extends beyond end of file.");
            return NULL;
        }

        record_contig = read_i32_le_raw(data, base);
        pos0 = read_i32_le_raw(data, base + 4);
        pos1 = (long long)pos0 + 1;
        if (record_contig == contig_id
            && (start_pos < 0 || pos1 >= start_pos)
            && (end_pos < 0 || pos1 <= end_pos)) {
            if (n_selected >= capacity) {
                if (capacity > PY_SSIZE_T_MAX / 2 || capacity * 2 > PY_SSIZE_T_MAX / 8) {
                    Py_DECREF(offsets_out);
                    PyBuffer_Release(&data_view);
                    PyErr_SetString(PyExc_MemoryError, "BCF selected-offset buffer is too large.");
                    return NULL;
                }
                capacity *= 2;
                if (PyByteArray_Resize(offsets_out, capacity * 8) < 0) {
                    Py_DECREF(offsets_out);
                    PyBuffer_Release(&data_view);
                    return NULL;
                }
            }
            write_i64_le(PyByteArray_AS_STRING(offsets_out) + n_selected * 8, (int64_t)offset);
            n_selected++;
        }

        offset = record_end;
        n_records++;
    }

    if (offset != data_len) {
        Py_DECREF(offsets_out);
        PyBuffer_Release(&data_view);
        PyErr_SetString(PyExc_ValueError, "Malformed BCF: record boundaries do not consume the full file.");
        return NULL;
    }
    if (PyByteArray_Resize(offsets_out, n_selected * 8) < 0) {
        Py_DECREF(offsets_out);
        PyBuffer_Release(&data_view);
        return NULL;
    }

    PyBuffer_Release(&data_view);
    return Py_BuildValue("Nnn", offsets_out, n_selected, n_records);
}

static PyMethodDef BcfMethods[] = {
    {"decode_gt", decode_gt, METH_VARARGS, "Decode full-file BCF FORMAT/GT records."},
    {"decode_core", decode_core, METH_VARARGS, "Decode full-file BCF core genotype and variant metadata records."},
    {"select_region_offsets", select_region_offsets, METH_VARARGS, "Select BCF record offsets by fixed CHROM/POS fields."},
    {NULL, NULL, 0, NULL}
};

static struct PyModuleDef BcfModule = {
    PyModuleDef_HEAD_INIT,
    "_bcf",
    NULL,
    -1,
    BcfMethods
};

PyMODINIT_FUNC
PyInit__bcf(void)
{
    return PyModule_Create(&BcfModule);
}
