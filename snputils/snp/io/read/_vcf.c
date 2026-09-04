#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <stdint.h>

#if defined(__unix__) || defined(__APPLE__)
#include <pthread.h>
#define VCF_HAVE_PTHREAD 1
#else
#define VCF_HAVE_PTHREAD 0
#endif

typedef struct {
    const unsigned char *data;
    const Py_ssize_t *sample_starts;
    const Py_ssize_t *sample_indices;
    Py_ssize_t start_record;
    Py_ssize_t end_record;
    Py_ssize_t n_samples;
    Py_ssize_t n_selected;
    Py_ssize_t row_width;
    int all_samples;
    int return_dosage;
    int failed;
    const char *error_message;
    char *out;
} VcfDosageTask;

static int
decode_allele(unsigned char value)
{
    if (value == '.' || value == '-') {
        return -1;
    }
    if (value < '0' || value > '9') {
        return -2;
    }
    return (int)(value - '0');
}

#if VCF_HAVE_PTHREAD
static void *
decode_vcf_gt_worker(void *argument)
{
    VcfDosageTask *task = (VcfDosageTask *)argument;

    for (Py_ssize_t record = task->start_record; record < task->end_record; record++) {
        const unsigned char *sample_row = task->data + task->sample_starts[record];
        char *out_row = task->out + record * task->row_width;

        for (Py_ssize_t out_sample = 0; out_sample < task->n_selected; out_sample++) {
            Py_ssize_t sample = task->all_samples ? out_sample : task->sample_indices[out_sample];
            const unsigned char *gt = sample_row + sample * 4;
            int first;
            int second;

            if (gt[1] != '|' && gt[1] != '/') {
                task->failed = 1;
                task->error_message = "The multithreaded VCF path requires diploid GT calls.";
                return NULL;
            }
            if (sample + 1 < task->n_samples && gt[3] != '\t') {
                task->failed = 1;
                task->error_message = "The multithreaded VCF path requires fixed-width GT-only sample fields.";
                return NULL;
            }
            first = decode_allele(gt[0]);
            second = decode_allele(gt[2]);
            if (first == -2 || second == -2) {
                task->failed = 1;
                task->error_message = "The multithreaded VCF path encountered an invalid GT allele.";
                return NULL;
            }
            if (task->return_dosage) {
                out_row[out_sample] = (char)((first < 0 || second < 0) ? -1 : first + second);
            } else {
                if (gt[1] != '|') {
                    task->failed = 1;
                    task->error_message =
                        "Cannot read unphased VCF genotypes with genotype_mode='phased'; "
                        "use genotype_mode='dosage' to load 0/1/2 genotype dosages.";
                    return NULL;
                }
                out_row[out_sample * 2] = (char)first;
                out_row[out_sample * 2 + 1] = (char)second;
            }
        }
    }
    return NULL;
}
#endif

static PyObject *
decode_gt(PyObject *self, PyObject *args)
{
    PyObject *data_obj = NULL;
    PyObject *sample_indices_obj = NULL;
    PyObject *sample_seq = NULL;
    Py_buffer data_view;
    const unsigned char *data;
    Py_ssize_t data_len;
    Py_ssize_t body_offset;
    Py_ssize_t n_samples;
    Py_ssize_t n_selected;
    Py_ssize_t *sample_indices = NULL;
    Py_ssize_t *sample_starts = NULL;
    int32_t *positions = NULL;
    Py_ssize_t capacity_records = 0;
    Py_ssize_t n_records = 0;
    Py_ssize_t offset;
    Py_ssize_t sample_bytes;
    PyObject *out = NULL;
    PyObject *positions_out = NULL;
    int all_samples;
    int return_dosage;
    int threads;

    data_view.buf = NULL;
    if (!PyArg_ParseTuple(
            args,
            "OnnOpi",
            &data_obj,
            &body_offset,
            &n_samples,
            &sample_indices_obj,
            &return_dosage,
            &threads)) {
        return NULL;
    }
    if (n_samples < 1) {
        PyErr_SetString(PyExc_ValueError, "Multithreaded VCF decoding requires at least one sample.");
        return NULL;
    }
    if (threads < 1) {
        PyErr_SetString(PyExc_ValueError, "VCF reader threads must be at least 1.");
        return NULL;
    }
    if (n_samples > (PY_SSIZE_T_MAX - 1) / 4) {
        PyErr_SetString(PyExc_MemoryError, "VCF genotype row is too large.");
        return NULL;
    }
    sample_bytes = n_samples * 4 - 1;

    all_samples = sample_indices_obj == Py_None;
    if (all_samples) {
        n_selected = n_samples;
    } else {
        sample_seq = PySequence_Fast(sample_indices_obj, "sample indices must be a sequence");
        if (sample_seq == NULL) {
            return NULL;
        }
        n_selected = PySequence_Fast_GET_SIZE(sample_seq);
        sample_indices = PyMem_New(Py_ssize_t, n_selected);
        if (n_selected > 0 && sample_indices == NULL) {
            Py_DECREF(sample_seq);
            PyErr_NoMemory();
            return NULL;
        }
        for (Py_ssize_t index = 0; index < n_selected; index++) {
            PyObject *item = PySequence_Fast_GET_ITEM(sample_seq, index);
            Py_ssize_t sample = PyLong_AsSsize_t(item);
            if (sample == -1 && PyErr_Occurred()) {
                Py_DECREF(sample_seq);
                PyMem_Free(sample_indices);
                return NULL;
            }
            if (sample < 0 || sample >= n_samples) {
                Py_DECREF(sample_seq);
                PyMem_Free(sample_indices);
                PyErr_SetString(PyExc_ValueError, "One or more VCF sample indexes are out of bounds.");
                return NULL;
            }
            sample_indices[index] = sample;
        }
        Py_DECREF(sample_seq);
        sample_seq = NULL;
    }

    if (PyObject_GetBuffer(data_obj, &data_view, PyBUF_SIMPLE) < 0) {
        PyMem_Free(sample_indices);
        return NULL;
    }
    data = (const unsigned char *)data_view.buf;
    data_len = data_view.len;
    if (body_offset < 0 || body_offset > data_len) {
        PyErr_SetString(PyExc_ValueError, "VCF body offset is out of bounds.");
        goto error;
    }

    capacity_records = (data_len - body_offset) / (sample_bytes + 32) + 1024;
    if (capacity_records > PY_SSIZE_T_MAX / (Py_ssize_t)sizeof(*sample_starts)) {
        PyErr_NoMemory();
        goto error;
    }
    sample_starts = PyMem_New(Py_ssize_t, capacity_records);
    if (capacity_records > 0 && sample_starts == NULL) {
        PyErr_NoMemory();
        goto error;
    }
    positions = PyMem_New(int32_t, capacity_records);
    if (capacity_records > 0 && positions == NULL) {
        PyErr_NoMemory();
        goto error;
    }

    offset = body_offset;
    while (offset < data_len) {
        Py_ssize_t tabs[9];
        Py_ssize_t cursor;
        Py_ssize_t content_end;
        Py_ssize_t record_end;
        int64_t position = 0;

        while (offset < data_len && (data[offset] == '\n' || data[offset] == '\r')) {
            offset++;
        }
        if (offset >= data_len) {
            break;
        }
        cursor = offset;
        for (int field = 0; field < 9; field++) {
            while (cursor < data_len && data[cursor] != '\t' && data[cursor] != '\n' && data[cursor] != '\r') {
                cursor++;
            }
            if (cursor >= data_len || data[cursor] != '\t') {
                PyErr_SetString(PyExc_ValueError, "VCF record has fewer than 10 tab-delimited fields.");
                goto error;
            }
            tabs[field] = cursor;
            cursor++;
        }

        if (tabs[8] - tabs[7] != 3 || data[tabs[7] + 1] != 'G' || data[tabs[7] + 2] != 'T') {
            PyErr_SetString(PyExc_ValueError, "The multithreaded VCF path requires FORMAT=GT.");
            goto error;
        }
        if (tabs[1] <= tabs[0] + 1) {
            PyErr_SetString(PyExc_ValueError, "VCF POS field is empty.");
            goto error;
        }
        for (Py_ssize_t pos_cursor = tabs[0] + 1; pos_cursor < tabs[1]; pos_cursor++) {
            unsigned char digit = data[pos_cursor];
            if (digit < '0' || digit > '9') {
                PyErr_SetString(PyExc_ValueError, "VCF POS field is not a positive integer.");
                goto error;
            }
            position = position * 10 + (int64_t)(digit - '0');
            if (position > INT32_MAX) {
                PyErr_SetString(PyExc_OverflowError, "VCF POS field exceeds int32 range.");
                goto error;
            }
        }
        if (return_dosage) {
            for (Py_ssize_t position = tabs[3] + 1; position < tabs[4]; position++) {
                if (data[position] != ',') {
                    continue;
                }
                PyErr_SetString(
                    PyExc_ValueError,
                    "genotype_mode='dosage' only supports biallelic variants; use genotype_mode='phased' for multiallelic allele calls.");
                goto error;
            }
        }

        if (sample_bytes > data_len - cursor) {
            PyErr_SetString(PyExc_ValueError, "VCF sample fields extend beyond end of file.");
            goto error;
        }
        content_end = cursor + sample_bytes;
        record_end = content_end;
        if (record_end < data_len && data[record_end] == '\r') {
            record_end++;
        }
        if (record_end < data_len) {
            if (data[record_end] != '\n') {
                PyErr_SetString(PyExc_ValueError, "The multithreaded VCF path requires fixed-width GT-only sample fields.");
                goto error;
            }
            record_end++;
        }
        for (Py_ssize_t sample = 0; sample < n_samples; sample++) {
            const unsigned char *gt = data + cursor + sample * 4;
            int first = decode_allele(gt[0]);
            int second = decode_allele(gt[2]);
            if ((gt[1] != '|' && gt[1] != '/')
                    || (sample + 1 < n_samples && gt[3] != '\t')) {
                PyErr_SetString(
                    PyExc_ValueError,
                    "The multithreaded VCF path requires fixed-width diploid GT-only sample fields.");
                goto error;
            }
            if (first == -2 || second == -2) {
                PyErr_SetString(PyExc_ValueError, "The multithreaded VCF path encountered an invalid GT allele.");
                goto error;
            }
        }

        if (n_records >= capacity_records) {
            Py_ssize_t new_capacity;
            Py_ssize_t *resized;
            int32_t *resized_positions;
            if (capacity_records > PY_SSIZE_T_MAX / 2) {
                PyErr_NoMemory();
                goto error;
            }
            new_capacity = capacity_records > 0 ? capacity_records * 2 : 1024;
            if (new_capacity > PY_SSIZE_T_MAX / (Py_ssize_t)sizeof(*sample_starts)) {
                PyErr_NoMemory();
                goto error;
            }
            resized = PyMem_Realloc(sample_starts, new_capacity * sizeof(*sample_starts));
            if (resized == NULL) {
                PyErr_NoMemory();
                goto error;
            }
            sample_starts = resized;
            resized_positions = PyMem_Realloc(positions, new_capacity * sizeof(*positions));
            if (resized_positions == NULL) {
                PyErr_NoMemory();
                goto error;
            }
            positions = resized_positions;
            capacity_records = new_capacity;
        }
        sample_starts[n_records] = cursor;
        positions[n_records] = (int32_t)position;
        n_records++;
        offset = record_end;
    }

    if (!return_dosage && n_selected > PY_SSIZE_T_MAX / 2) {
        PyErr_SetString(PyExc_MemoryError, "VCF genotype row is too large.");
        goto error;
    }
    Py_ssize_t row_width = return_dosage ? n_selected : n_selected * 2;
    if (row_width != 0 && n_records > PY_SSIZE_T_MAX / row_width) {
        PyErr_SetString(PyExc_MemoryError, "VCF genotype output is too large.");
        goto error;
    }
    out = PyByteArray_FromStringAndSize(NULL, n_records * row_width);
    if (out == NULL) {
        goto error;
    }
    positions_out = PyByteArray_FromStringAndSize(
        (const char *)positions,
        n_records * (Py_ssize_t)sizeof(*positions));
    if (positions_out == NULL) {
        goto error;
    }

#if VCF_HAVE_PTHREAD
    if (n_records > 0 && n_selected > 0) {
        Py_ssize_t thread_count = threads;
        Py_ssize_t created_threads = 0;
        int create_failed = 0;
        VcfDosageTask *tasks;
        pthread_t *thread_ids;

        if (thread_count > n_records) {
            thread_count = n_records;
        }
        tasks = PyMem_Calloc((size_t)thread_count, sizeof(*tasks));
        thread_ids = PyMem_New(pthread_t, thread_count);
        if (tasks == NULL || thread_ids == NULL) {
            PyMem_Free(tasks);
            PyMem_Free(thread_ids);
            PyErr_NoMemory();
            goto error;
        }
        for (Py_ssize_t thread = 0; thread < thread_count; thread++) {
            VcfDosageTask *task = &tasks[thread];
            task->data = data;
            task->sample_starts = sample_starts;
            task->sample_indices = sample_indices;
            task->start_record = (thread * n_records) / thread_count;
            task->end_record = ((thread + 1) * n_records) / thread_count;
            task->n_samples = n_samples;
            task->n_selected = n_selected;
            task->row_width = row_width;
            task->all_samples = all_samples;
            task->return_dosage = return_dosage;
            task->out = PyByteArray_AS_STRING(out);
        }

        Py_BEGIN_ALLOW_THREADS
        for (Py_ssize_t thread = 0; thread < thread_count; thread++) {
            if (pthread_create(&thread_ids[thread], NULL, decode_vcf_gt_worker, &tasks[thread]) != 0) {
                create_failed = 1;
                break;
            }
            created_threads++;
        }
        for (Py_ssize_t thread = 0; thread < created_threads; thread++) {
            pthread_join(thread_ids[thread], NULL);
        }
        Py_END_ALLOW_THREADS

        if (create_failed) {
            PyMem_Free(thread_ids);
            PyMem_Free(tasks);
            PyErr_SetString(PyExc_RuntimeError, "Could not create the requested VCF reader threads.");
            goto error;
        }
        for (Py_ssize_t thread = 0; thread < thread_count; thread++) {
            if (tasks[thread].failed) {
                const char *error_message = tasks[thread].error_message;
                PyMem_Free(thread_ids);
                PyMem_Free(tasks);
                PyErr_SetString(PyExc_ValueError, error_message);
                goto error;
            }
        }
        PyMem_Free(thread_ids);
        PyMem_Free(tasks);
    }
#else
    if (threads > 1) {
        PyErr_SetString(PyExc_NotImplementedError, "Multithreaded VCF reading is unavailable on this platform.");
        goto error;
    }
#endif

    PyMem_Free(sample_starts);
    PyMem_Free(positions);
    PyMem_Free(sample_indices);
    PyBuffer_Release(&data_view);
    return Py_BuildValue("NNn", out, positions_out, n_records);

error:
    Py_XDECREF(out);
    Py_XDECREF(positions_out);
    PyMem_Free(sample_starts);
    PyMem_Free(positions);
    PyMem_Free(sample_indices);
    if (data_view.buf != NULL) {
        PyBuffer_Release(&data_view);
    }
    return NULL;
}

static PyMethodDef VcfMethods[] = {
    {
        "decode_gt",
        decode_gt,
        METH_VARARGS,
        "Decode fixed-width GT-only VCF records into an ordered int8 genotype array."
    },
    {NULL, NULL, 0, NULL}
};

static struct PyModuleDef vcfmodule = {
    PyModuleDef_HEAD_INIT,
    "_vcf",
    NULL,
    -1,
    VcfMethods
};

PyMODINIT_FUNC
PyInit__vcf(void)
{
    return PyModule_Create(&vcfmodule);
}
