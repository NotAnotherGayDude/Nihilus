#include <string.h>
#include <stdlib.h>
#include <stdint.h>
#include <stdbool.h>
#include <stdio.h>
#include <sys/stat.h>

// Constants
#define TENSOR_LEN_SIZE sizeof(uint64_t)
#define NAME_LEN_SIZE sizeof(uint64_t)
#define DIMS_SIZE (sizeof(size_t) * 4)
#define TYPE_SIZE sizeof(enum ggml_type)
#define OP_SIZE sizeof(enum ggml_op)
#define GGML_MAX_SRC 10

struct name_data {
	size_t length;
	const char* data;
};

typedef struct {
	size_t dims[4];
	enum ggml_type type;
	enum ggml_op op;
	struct name_data name_val;
	struct name_data* input_names;
	size_t input_names_count;
} intermediary_tensor;

// Helper function to copy a string and create name_data
struct name_data create_name_data(const char* source) {
	struct name_data result = { 0, NULL };

	if (!source) {
		return result;
	}

	size_t len = strlen(source);
	char* copy = ( char* )malloc(len + 1);
	if (!copy) {
		return result;
	}

	strcpy(copy, source);
	result.length = len;
	result.data	  = copy;

	return result;
}

// Free name_data memory
void free_name_data(struct name_data* name) {
	if (name && name->data) {
		free(( void* )name->data);
		name->data	 = NULL;
		name->length = 0;
	}
}

// Create intermediary_tensor from ggml_tensor
intermediary_tensor* create_intermediary_from_ggml(const struct ggml_tensor* ggml_tensor) {
	if (!ggml_tensor) {
		return NULL;
	}

	// Allocate the intermediary tensor
	intermediary_tensor* tensor = ( intermediary_tensor* )malloc(sizeof(intermediary_tensor));
	if (!tensor) {
		return NULL;
	}

	// Initialize to zero
	memset(tensor, 0, sizeof(intermediary_tensor));

	// Copy dimensions
	for (size_t i = 0; i < 4; ++i) {
		tensor->dims[i] = ggml_tensor->ne[i];
	}

	// Copy type and operation
	tensor->type = ggml_tensor->type;
	tensor->op	 = ggml_tensor->op;

	// Copy tensor name
	tensor->name_val = create_name_data(ggml_tensor->name);
	if (ggml_tensor->name && !tensor->name_val.data) {
		// Failed to allocate name
		free(tensor);
		return NULL;
	}

	// Count valid source tensors
	size_t src_count = 0;
	for (int i = 0; i < GGML_MAX_SRC; ++i) {
		if (ggml_tensor->src[i] && ggml_tensor->src[i]->name) {
			src_count++;
		}
	}

	tensor->input_names_count = src_count;

	// If no sources, we're done
	if (src_count == 0) {
		tensor->input_names = NULL;
		return tensor;
	}

	// Allocate input names array
	tensor->input_names = malloc(src_count * sizeof(struct name_data));
	if (!tensor->input_names) {
		free_name_data(&tensor->name_val);
		free(tensor);
		return NULL;
	}

	// Copy source names
	size_t name_index = 0;
	for (int i = 0; i < GGML_MAX_SRC; ++i) {
		if (ggml_tensor->src[i] && ggml_tensor->src[i]->name) {
			tensor->input_names[name_index] = create_name_data(ggml_tensor->src[i]->name);

			if (!tensor->input_names[name_index].data) {
				// Cleanup on failure
				for (size_t j = 0; j < name_index; ++j) {
					free_name_data(&tensor->input_names[j]);
				}
				free(tensor->input_names);
				free_name_data(&tensor->name_val);
				free(tensor);
				return NULL;
			}

			name_index++;
		}
	}

	return tensor;
}

// Free intermediary_tensor
void free_intermediary_tensor(intermediary_tensor* tensor) {
	if (!tensor) {
		return;
	}

	// Free name
	free_name_data(&tensor->name_val);

	// Free input names
	if (tensor->input_names) {
		for (size_t i = 0; i < tensor->input_names_count; ++i) {
			free_name_data(&tensor->input_names[i]);
		}
		free(tensor->input_names);
	}

	free(tensor);
}

// Serialization structure
typedef struct {
	uint8_t* data;
	size_t size;
} serialized_tensor;

// Calculate total size needed for serialization
size_t calculate_serialized_size(const intermediary_tensor* tensor) {
	if (!tensor) {
		return 0;
	}

	size_t total = TYPE_SIZE + OP_SIZE + DIMS_SIZE + NAME_LEN_SIZE;// Basic header (now includes op)
	total += tensor->name_val.length;// Tensor name data
	total += sizeof(uint64_t);// Input names count

	// Input names sizes
	for (size_t i = 0; i < tensor->input_names_count; ++i) {
		total += NAME_LEN_SIZE;// Each input name length
		total += tensor->input_names[i].length;// Each input name data
	}

	return total;
}

// Serialize intermediary_tensor to binary format
serialized_tensor serialize_intermediary_tensor(const intermediary_tensor* tensor) {
	serialized_tensor result = { NULL, 0 };

	if (!tensor) {
		return result;
	}

	size_t total_size = calculate_serialized_size(tensor);
	uint8_t* buffer	  = ( uint8_t* )malloc(total_size);
	if (!buffer) {
		return result;
	}

	size_t offset = 0;

	// Serialize type
	memcpy(buffer + offset, &tensor->type, TYPE_SIZE);
	offset += TYPE_SIZE;

	// Serialize operation
	memcpy(buffer + offset, &tensor->op, OP_SIZE);
	offset += OP_SIZE;

	// Serialize dimensions
	memcpy(buffer + offset, tensor->dims, DIMS_SIZE);
	offset += DIMS_SIZE;

	// Serialize tensor name length and data
	uint64_t name_len = tensor->name_val.length;
	memcpy(buffer + offset, &name_len, NAME_LEN_SIZE);
	offset += NAME_LEN_SIZE;

	if (name_len > 0) {
		memcpy(buffer + offset, tensor->name_val.data, name_len);
		offset += name_len;
	}

	// Serialize input names count
	uint64_t input_count = tensor->input_names_count;
	memcpy(buffer + offset, &input_count, sizeof(uint64_t));
	offset += sizeof(uint64_t);

	// Serialize each input name
	for (size_t i = 0; i < tensor->input_names_count; ++i) {
		uint64_t input_name_len = tensor->input_names[i].length;
		memcpy(buffer + offset, &input_name_len, NAME_LEN_SIZE);
		offset += NAME_LEN_SIZE;

		if (input_name_len > 0) {
			memcpy(buffer + offset, tensor->input_names[i].data, input_name_len);
			offset += input_name_len;
		}
	}

	result.data = buffer;
	result.size = total_size;
	return result;
}

// Free serialized tensor
void free_serialized_tensor(serialized_tensor* tensor) {
	if (tensor && tensor->data) {
		free(tensor->data);
		tensor->data = NULL;
		tensor->size = 0;
	}
}

// Check if file exists
bool file_exists(const char* filename) {
	if (!filename) {
		return false;
	}

	struct stat buffer;
	return (stat(filename, &buffer) == 0);
}

// Write binary data to file
bool write_file(const char* filename, const void* data, size_t size) {
	if (!filename || !data || size == 0) {
		return false;
	}

	FILE* file = fopen(filename, "wb");
	if (!file) {
		return false;
	}

	size_t written = fwrite(data, 1, size, file);
	bool success   = (written == size);

	fclose(file);
	return success;
}

// Main function: Check if .safetensor file exists, if not serialize and save
// Main function: Check if .safetensor file exists, if not serialize and save
// Main function: Check if .safetensor file exists, if not serialize and save
bool save_tensor_if_not_exists(const struct ggml_tensor* tensor_new) {
	intermediary_tensor* tensor = create_intermediary_from_ggml(tensor_new);

	if (!tensor) {
		printf("DEBUG: tensor is NULL\n");
		return false;
	}

	// Check if length is reasonable (probably corrupted if > 1MB)
	if (tensor->name_val.length > 1000000) {
		printf("DEBUG: tensor name length is suspiciously large: %zu - probably corrupted\n", tensor->name_val.length);
		free_intermediary_tensor(tensor);
		return false;
	}

	if (!tensor->name_val.data) {
		printf("DEBUG: tensor name data is NULL\n");
		free_intermediary_tensor(tensor);
		return false;
	}

	if (tensor->name_val.length == 0) {
		printf("DEBUG: tensor name length is 0\n");
		free_intermediary_tensor(tensor);
		return false;
	}

	// Sanitize filename - remove any problematic characters
	char* safe_name = ( char* )malloc(tensor->name_val.length + 1);
	if (!safe_name) {
		printf("DEBUG: malloc failed for size %zu\n", tensor->name_val.length + 1);
		free_intermediary_tensor(tensor);
		return false;
	}

	// Copy name and replace problematic chars
	for (size_t i = 0; i < tensor->name_val.length; ++i) {
		char c = tensor->name_val.data[i];
		if (c == '/' || c == '\\' || c == ':' || c == '*' || c == '?' || c == '"' || c == '<' || c == '>' || c == '|') {
			safe_name[i] = '_';
		} else {
			safe_name[i] = c;
		}
	}
	safe_name[tensor->name_val.length] = '\0';

	// Check if tensor name contains "Qcur"
	bool has_qcur = (strstr(safe_name, "Qcur") != NULL);

	if (has_qcur) {
		const char* op_name = ggml_op_name(tensor->op);
	}

	// Create filename: tensor_name + ".safetensor" (with kernel_type if contains "Qcur")
	size_t filename_len;
	char* filename;

	if (has_qcur) {
		const char* op_name = ggml_op_name(tensor->op);
		filename_len		= strlen(safe_name) + 1 + strlen(op_name) + strlen(".safetensor") + 1;
		filename			= ( char* )malloc(filename_len);
		if (!filename) {
			printf("DEBUG: failed to allocate filename with kernel_type\n");
			free(safe_name);
			free_intermediary_tensor(tensor);
			return false;
		}
		sprintf(filename, "%s_%s.safetensor", safe_name, op_name);
	} else {
		filename_len = strlen(safe_name) + strlen(".safetensor") + 1;
		filename	 = ( char* )malloc(filename_len);
		if (!filename) {
			printf("DEBUG: failed to allocate filename\n");
			free(safe_name);
			free_intermediary_tensor(tensor);
			return false;
		}
		strcpy(filename, safe_name);
		strcat(filename, ".safetensor");
	}

	// Create txt filename with same logic
	size_t txt_filename_len;
	char* txt_filename;

	if (has_qcur) {
		const char* op_name = ggml_op_name(tensor->op);
		txt_filename_len	= strlen(safe_name) + 1 + strlen(op_name) + strlen(".txt") + 1;
		txt_filename		= ( char* )malloc(txt_filename_len);
		if (!txt_filename) {
			printf("DEBUG: failed to allocate txt filename with kernel_type\n");
			free(filename);
			free(safe_name);
			free_intermediary_tensor(tensor);
			return false;
		}
		sprintf(txt_filename, "%s_%s.txt", safe_name, op_name);
	} else {
		txt_filename_len = strlen(safe_name) + strlen(".txt") + 1;
		txt_filename	 = ( char* )malloc(txt_filename_len);
		if (!txt_filename) {
			printf("DEBUG: failed to allocate txt filename\n");
			free(filename);
			free(safe_name);
			free_intermediary_tensor(tensor);
			return false;
		}
		strcpy(txt_filename, safe_name);
		strcat(txt_filename, ".txt");
	}

	// Check if files already exist
	bool safetensor_exists = file_exists(filename);
	bool txt_exists		   = file_exists(txt_filename);

	if (safetensor_exists && txt_exists) {
		//printf("Files %s and %s already exist, skipping...\n", filename, txt_filename);
		free(txt_filename);
		free(filename);
		free(safe_name);
		free_intermediary_tensor(tensor);
		return true;// Success - we "saved" by not overwriting
	}

	// Serialize the tensor (needed for binary file)
	serialized_tensor serialized = { NULL, 0 };
	bool need_binary			 = !safetensor_exists;

	if (need_binary) {
		serialized = serialize_intermediary_tensor(tensor);
		if (!serialized.data) {
			printf("Failed to serialize tensor %s\n", safe_name);
			free(txt_filename);
			free(filename);
			free(safe_name);
			free_intermediary_tensor(tensor);
			return false;
		}
	}

	bool success = true;

	// Write binary file if needed
	if (need_binary) {
		success = write_file(filename, serialized.data, serialized.size);
		if (!success) {
			printf("Failed to write file %s\n", filename);
		}
	}

	// Write text file if needed
	if (!txt_exists) {
		FILE* txt_file = fopen(txt_filename, "w");
		if (txt_file) {
			fprintf(txt_file, "Tensor: \"%.*s\"\n", ( int )tensor->name_val.length, tensor->name_val.data);
			fprintf(txt_file, "  Dimensions: [%zu, %zu, %zu, %zu]\n", tensor->dims[0], tensor->dims[1], tensor->dims[2], tensor->dims[3]);
			fprintf(txt_file, "  Type: %s\n", ggml_type_name(tensor->type));
			fprintf(txt_file, "  Operation: %s\n", ggml_op_name(tensor->op));
			fprintf(txt_file, "  Input Names Count: %zu\n", tensor->input_names_count);

			for (size_t i = 0; i < tensor->input_names_count; ++i) {
				fprintf(txt_file, "    Input %zu: \"%.*s\"\n", i, ( int )tensor->input_names[i].length, tensor->input_names[i].data);
			}

			fclose(txt_file);
		} else {
			printf("Failed to create txt file %s\n", txt_filename);
		}
	}

	// Cleanup
	if (need_binary) {
		free_serialized_tensor(&serialized);
	}
	free(txt_filename);
	free(filename);
	free(safe_name);
	free_intermediary_tensor(tensor);

	return success;
}