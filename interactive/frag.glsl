#version 460 core

layout (binding = 0) uniform sampler2D text;

in vec2 frag_uv;
out vec4 color;

void main() {
    color = vec4(1.0 - texture(text, vec2(frag_uv.x, 1.0 - frag_uv.y)).rrr, 1.0f);
}
