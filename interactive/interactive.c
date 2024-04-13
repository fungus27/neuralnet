#include <stdlib.h>
#include <stdio.h>
#include <assert.h>
#include <string.h>
#include <errno.h>
#include <math.h>
#include <GL/glew.h>
#include <GLFW/glfw3.h>

#include "../neural_network.h"

#define SWAP32(num) (((num)>>24)&0xff) | (((num)<<8)&0xff0000) | (((num)>>8)&0xff00) | (((num)<<24)&0xff000000)
#define MAX(x, y) ( (x) > (y) ? (x) : (y) )
#define MIN(x, y) ( (x) < (y) ? (x) : (y) )

#define INBOUNDS(x, y, w, h) ((x) >= 0 && (x) < (w) && (y) >= 0 && (y) < (h))

const unsigned int width = 560, height = 560;
float *canvas;


neural_network nn;

void error_callback(int error, const char* description) {
    fprintf(stderr, "glfw error: %s\n", description);
    exit(EXIT_FAILURE);
}

unsigned int compile_render_shaders(const char *vert_filepath, const char *frag_filepath) {
    FILE *vert_file = fopen(vert_filepath, "r");
    if (!vert_file) {
        int error = errno;
        fprintf(stderr, "unable to load vertex shader at '%s'.\nerror: %s", vert_filepath, strerror(error));
        exit(EXIT_FAILURE);
    }

    FILE *frag_file = fopen(frag_filepath, "r");
    if (!frag_file) {
        int error = errno;
        fprintf(stderr, "unable to load fragment shader at '%s'.\nerror: %s", frag_filepath, strerror(error));
        exit(EXIT_FAILURE);
    }

    fseek(vert_file, 0, SEEK_END);
    unsigned int vert_size = ftell(vert_file);
    fseek(vert_file, 0, SEEK_SET);

    fseek(frag_file, 0, SEEK_END);
    unsigned int frag_size = ftell(frag_file);
    fseek(frag_file, 0, SEEK_SET);

    char *vert_content = malloc(vert_size + 1);
    fread(vert_content, vert_size, 1, vert_file);
    vert_content[vert_size] = 0;

    char *frag_content = malloc(frag_size + 1);
    fread(frag_content, frag_size, 1, frag_file);
    frag_content[frag_size] = 0;

    unsigned int vert = glCreateShader(GL_VERTEX_SHADER);
    glShaderSource(vert, 1, (const char**)&vert_content, NULL);
    glCompileShader(vert);

    int success;
    char infoLog[512];
    glGetShaderiv(vert, GL_COMPILE_STATUS, &success);

    if (!success) {
        glGetShaderInfoLog(vert, 512, NULL, infoLog);
        fprintf(stderr, "vertex shader failed to compile.\nerror: %s\n", infoLog);
        exit(EXIT_FAILURE);
    }

    unsigned int frag = glCreateShader(GL_FRAGMENT_SHADER);
    glShaderSource(frag, 1, (const char**)&frag_content, NULL);
    glCompileShader(frag);

    glGetShaderiv(frag, GL_COMPILE_STATUS, &success);
    if (!success) {
        glGetShaderInfoLog(frag, 512, NULL, infoLog);
        fprintf(stderr, "fragment shader failed to compile.\nerror: %s\n", infoLog);
        exit(EXIT_FAILURE);
    }
    
    free(vert_content);
    free(frag_content);

    unsigned int prog = glCreateProgram();

    glAttachShader(prog, vert);
    glAttachShader(prog, frag);

    glLinkProgram(prog);

    glGetProgramiv(prog, GL_LINK_STATUS, &success);
    if(!success) {
        glGetProgramInfoLog(prog, 512, NULL, infoLog);
        fprintf(stderr, "render shader program failed to link.\nerror: %s\n", infoLog);
        exit(EXIT_FAILURE);
    }

    glDeleteShader(vert);
    glDeleteShader(frag);

    return prog;
}

unsigned int compile_compute_shader(const char* filepath) {
    FILE *file = fopen(filepath, "r");
    if (!file) {
        int error = errno;
        fprintf(stderr, "unable to load compute shader at '%s'.\nerror: %s", filepath, strerror(error));
        exit(EXIT_FAILURE);
    }

    fseek(file, 0, SEEK_END);
    unsigned int size = ftell(file);
    fseek(file, 0, SEEK_SET);

    char *content = malloc(size + 1);
    fread(content, size, 1, file);
    content[size] = 0;

    unsigned int shader = glCreateShader(GL_COMPUTE_SHADER);
    glShaderSource(shader, 1, (const char**)&content, NULL);
    glCompileShader(shader);

    int success;
    char infoLog[512];
    glGetShaderiv(shader, GL_COMPILE_STATUS, &success);

    if (!success) {
        glGetShaderInfoLog(shader, 512, NULL, infoLog);
        fprintf(stderr, "compute shader failed to compile.\nerror: %s\n", infoLog);
        exit(EXIT_FAILURE);
    }

    free(content);

    unsigned int prog = glCreateProgram();

    glAttachShader(prog, shader);

    glLinkProgram(prog);
    
    glGetProgramiv(prog, GL_LINK_STATUS, &success);
    if(!success) {
        glGetProgramInfoLog(prog, 512, NULL, infoLog);
        fprintf(stderr, "compute shader program failed to link.\nerror: %s\n", infoLog);
        exit(EXIT_FAILURE);
    }

    glDeleteShader(shader);

    return prog;
}

void draw(float *texture, float x, float y, float strength) {
    const float a = 0.88622692f;
    float r = sqrt(-log(1.0f/255.0f) * strength) + 1.0f;
    float cell_x = floor(x) - 0.5f, cell_y = floor(27.0f - y) + 0.5f;
    int bl_corner_x = (int)floor(cell_x - r), bl_corner_y = (int)ceil(cell_y - r);
    int tr_corner_x = (int)floor(cell_x + r), tr_corner_y = (int)ceil(cell_y + r);  


    for (int ix = bl_corner_x; ix <= tr_corner_x; ++ix) {
        for (int iy = bl_corner_y; iy <= tr_corner_y; ++iy) {
            if (INBOUNDS(ix, iy, 28, 28)) {
                float rel_x = ((float)ix - x)/strength;
                float rel_y = ((float)(27 - iy) - y)/strength;
                texture[iy * 28 + ix] += a*a * (erf(rel_x + 1.0) - erf(rel_x)) * (erf(rel_y + 1.0) - erf(rel_y));
                //texture[iy * 28 + ix] = 0.5f;
            }
        }
    }
}

void mouse_button_callback(GLFWwindow* window, int button, int action, int mods) {
    if (button == GLFW_MOUSE_BUTTON_LEFT && action == GLFW_PRESS) {
        double x, y;
        glfwGetCursorPos(window, &x, &y);
        x = x/width * 28.0;
        y = (height - y)/height * 28.0;
        //printf("%4.2f %4.2f\n", x, y);
        
        draw(canvas, x, y, 1.2f);
        glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, 28, 28, GL_RED, GL_FLOAT, canvas);

        float out[10];
        evaluate_nn(&nn, canvas, out);
        float max = out[0];
        unsigned int max_index = 0;
        for (unsigned int i = 0; i < 10; ++i) {
            //printf("%u: %4.2f ", i, out[i]);
            if (out[i] > max) {
                max = out[i];
                max_index = i;
            }
        }
        printf("current prediction: %u\n", max_index);
        //printf("\n");
    }
}

void read_data(char *images_name, char *labels_name, float *inputs, float *expected_outputs, unsigned char *labels) {
    
    {
        FILE *f_images = fopen(images_name, "r");

        int magic_number; 
        fread(&magic_number, sizeof(int), 1, f_images);
        magic_number = SWAP32(magic_number);
        assert(magic_number == 2051);

        int number_of_images;
        fread(&number_of_images, sizeof(int), 1, f_images);
        number_of_images = SWAP32(number_of_images);
        //assert(number_of_images == 60000);

        int rows, cols;
        fread(&rows, sizeof(int), 1, f_images);
        fread(&cols, sizeof(int), 1, f_images);
        rows = SWAP32(rows);
        cols = SWAP32(cols);
        assert(rows == 28 && cols == 28);

        for (unsigned int i = 0; i < number_of_images; ++i) {
            for (unsigned int j = 0; j < rows * cols; ++j) {
                unsigned char pixel;
                fread(&pixel, sizeof(unsigned char), 1, f_images);
                inputs[i * rows * cols + j] = pixel/255.0f;
            }
        }

        fclose(f_images);
    }

    {
        FILE *f_labels = fopen(labels_name, "r");

        int magic_number; 
        fread(&magic_number, sizeof(int), 1, f_labels);
        magic_number = SWAP32(magic_number);
        assert(magic_number == 2049);

        int number_of_labels;
        fread(&number_of_labels, sizeof(int), 1, f_labels);
        number_of_labels = SWAP32(number_of_labels);
        //assert(number_of_labels == 60000);

        for (unsigned int i = 0; i < number_of_labels; ++i) {
            fread(&labels[i], sizeof(unsigned char), 1, f_labels);

            if (expected_outputs) {
                for (unsigned int j = 0; j < 10; ++j)
                    expected_outputs[i * 10 + j] = (j == labels[i]) ? 1.0f : 0.0f;
            }

        }

        fclose(f_labels);
    }
}

int main() {
    if (!glfwInit()) {
        const char *description;
        int code = glfwGetError(&description);
        fprintf(stderr, "glfw failed to initialize.\nerror (%d): %s\n", code, description);
        exit(EXIT_FAILURE);
    }


    glfwSetErrorCallback(error_callback);
    glfwWindowHint(GLFW_RESIZABLE, GLFW_FALSE);
    GLFWwindow* window = glfwCreateWindow(width, height, "nn", NULL, NULL);
    glfwMakeContextCurrent(window);
    glfwSetMouseButtonCallback(window, mouse_button_callback);

    GLenum err; 
    if ((err = glewInit()) != GLEW_OK) {
        fprintf(stderr, "glew failed to initialize.\nerror (%u): %s\n", err, glewGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    printf("version: %s\n", glGetString(GL_VERSION));

    float vertices[] = {
        1.0f, 1.0f, 1.0f, 1.0f,
        1.0f, -1.0f, 1.0f, 0.0f,
        -1.0f, -1.0f, 0.0f, 0.0f,
        -1.0f, 1.0f, 0.0f, 1.0f
    };

    unsigned int indices[] = {0, 1, 2, 2, 3, 0};

    unsigned int vbo;
    glGenBuffers(1, &vbo);
    glBindBuffer(GL_ARRAY_BUFFER, vbo);
    glBufferData(GL_ARRAY_BUFFER, sizeof(vertices), vertices, GL_STATIC_DRAW);

    unsigned int vao;
    glGenVertexArrays(1, &vao);
    glBindVertexArray(vao);

    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 4 * sizeof(float), (void*)0);
    glEnableVertexAttribArray(0);

    glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 4 * sizeof(float), (void*)(2 * sizeof(float)));
    glEnableVertexAttribArray(1);

    unsigned int ebo;
    glGenBuffers(1, &ebo);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ebo);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(indices), indices, GL_STATIC_DRAW);

    float *images = malloc(10000 * 28 * 28 * sizeof(float));
    unsigned char *labels = malloc(10000 * sizeof(char));
    read_data("../mnist/t10k-images-idx3-ubyte", "../mnist/t10k-labels-idx1-ubyte", images, NULL, labels);

    unsigned int index = 1536;
    float *image = images + index * 28 * 28;
    unsigned char label = labels[index];

    FILE *saved = fopen("../models/sigmoid_number_classifier.nn", "r");
    load_nn(&nn, saved);
    fclose(saved);
    float nn_out[10];
    evaluate_nn(&nn, image, nn_out);
    float max = nn_out[0];
    unsigned int imax = 0;
    for (unsigned int i = 1; i < 10; ++i) {
        if (nn_out[i] > max) {
            max = nn_out[i];
            imax = i;
        }
    }

    canvas = calloc(28 * 28, sizeof(float));
    //draw(canvas, 14.5f, 18.0f, 2.0f);

    unsigned int texture;
    glGenBuffers(1, &texture);
    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, texture);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_R32F, 28, 28, 0, GL_RED, GL_FLOAT, canvas);
    //printf("label: %u\npredicted: %u\n", label, imax);

    unsigned int render_prog = compile_render_shaders("vert.glsl", "frag.glsl");
    glUseProgram(render_prog);

    while (!glfwWindowShouldClose(window)) {
        glClear(GL_COLOR_BUFFER_BIT);

        glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_INT, 0);

        glfwSwapBuffers(window);
        glfwPollEvents();
    }
    free(images);
    free(labels);
    glfwTerminate();
}
