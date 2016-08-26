
-- xforms

uniform mat4 Projection;
uniform mat4 Modelview;
uniform mat3 NormalMatrix;

-- vert

in vec4 Position;
in vec3 Normal;
out vec4 vPosition;
out vec3 vNormal;
void main()
{
    vPosition = Position;
    vNormal = Normal;
    gl_Position = Projection * Modelview * Position;
}

-- geom

in vec4 vPosition[4];
in vec3 vNormal[4];
out vec3 gNormal;
const bool facets = true;

layout(lines_adjacency) in;
layout(triangle_strip, max_vertices = 4) out;

void emit(int index)
{
    if (!facets) {
        gNormal = NormalMatrix * vNormal[index];
    }
    gl_Position = Projection * Modelview * vPosition[index];
    EmitVertex(); 
}

void main()
{
    if (facets) {
        vec3 A = vPosition[0].xyz;
        vec3 B = vPosition[1].xyz;
        vec3 C = vPosition[2].xyz;
        gNormal = NormalMatrix * normalize(cross(B - A, C - A));
    }

    emit(0);
    emit(1);
    emit(3);
    emit(2);
    EndPrimitive();
}

-- frag

in vec3 gNormal;
out vec4 FragColor;

uniform vec4 LightPosition = vec4(0.75, -0.25, 1, 1);
uniform vec3 AmbientMaterial = vec3(0.2, 0.1, 0.1);
uniform vec4 DiffuseMaterial = vec4(1.0, 209.0/255.0, 54.0/255.0, 1.0);
uniform vec3 SpecularMaterial = vec3(0.4, 0.4, 0.3);
uniform float Shininess = 200.0;
uniform float Fresnel = 0.1;

void main()
{
    vec3 N = normalize(gNormal);
    vec3 L = normalize((LightPosition).xyz);

    vec3 Eye = vec3(0, 0, 1);
    vec3 H = normalize(L + Eye);

    float df = max(0.0, dot(N, L));
    float sf = pow(max(0.0, dot(N, H)), Shininess);
    float rfTheta = Fresnel + (1-Fresnel) * pow(1-dot(N,Eye), 5);

    vec3 color = AmbientMaterial +
        df * DiffuseMaterial.rgb +
        sf * SpecularMaterial +
        rfTheta;

    FragColor = vec4(color, DiffuseMaterial.a);
}
