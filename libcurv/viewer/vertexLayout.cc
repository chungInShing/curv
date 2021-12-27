#include "vertexLayout.h"
#include "text.h"
#include <libcurv/dtostr.h>

std::map<GLint, GLuint> VertexLayout::s_enabledAttribs = std::map<GLint, GLuint>();

VertexLayout::VertexLayout(std::vector<VertexAttrib> _attribs) : m_attribs(_attribs), m_stride(0), m_positionAttribIndex(-1), m_colorAttribIndex(-1), m_normalAttribIndex(-1), m_texCoordAttribIndex(-1) {

    m_stride = 0;
    for (unsigned int i = 0; i < m_attribs.size(); i++) {

        // Set the offset of this vertex attribute: The stride at this point denotes the number
        // of bytes into the vertex by which this attribute is offset, but we must cast the number
        // as a void* to use with glVertexAttribPointer; We use reinterpret_cast to avoid warnings
        m_attribs[i].offset = reinterpret_cast<void*>(m_stride);

        GLint byteSize = m_attribs[i].size;

        switch (m_attribs[i].type) {
            case GL_FLOAT:
            case GL_INT:
            case GL_UNSIGNED_INT:
                byteSize *= 4; // 4 bytes for floats, ints, and uints
                break;
            case GL_SHORT:
            case GL_UNSIGNED_SHORT:
                byteSize *= 2; // 2 bytes for shorts and ushorts
                break;
        }

        if ( m_attribs[i].attrType == POSITION_ATTRIBUTE ){
            m_positionAttribIndex = i;
        }
        else if ( m_attribs[i].attrType == COLOR_ATTRIBUTE ){
            m_colorAttribIndex = i;
        }
        else if ( m_attribs[i].attrType == NORMAL_ATTRIBUTE ){
            m_normalAttribIndex = i;
        }
        else if ( m_attribs[i].attrType == TEXCOORD_ATTRIBUTE ){
            m_texCoordAttribIndex = i;
        }

        m_stride += byteSize;

        // TODO: Automatically add padding or warn if attributes are not byte-aligned
    }
}

VertexLayout::~VertexLayout() {
    m_attribs.clear();
}

void VertexLayout::enable(const Shader* _program) {
    GLuint glProgram = _program->getProgram();

    // Enable all attributes for this layout
    for (unsigned int i = 0; i < m_attribs.size(); i++) {
        const GLint location = _program->getAttribLocation("a_"+m_attribs[i].name);
        if (location != -1) {
            glEnableVertexAttribArray(location);
            glVertexAttribPointer(location, m_attribs[i].size, m_attribs[i].type, m_attribs[i].normalized, m_stride, m_attribs[i].offset);
            s_enabledAttribs[location] = glProgram; // Track currently enabled attribs by the program to which they are bound
        }
    }

    // Disable previously bound and now-unneeded attributes
    for (std::map<GLint, GLuint>::iterator it=s_enabledAttribs.begin(); it!=s_enabledAttribs.end(); ++it){
        const GLint& location = it->first;
        GLuint& boundProgram = it->second;

        if (boundProgram != glProgram && boundProgram != 0) {
            glDisableVertexAttribArray(location);
            boundProgram = 0;
        }
    }
}

#ifdef MULTIPASS_RENDER

std::string VertexLayout::getDefaultFPVertShader(std::string bbox) {
    std::string rta =
"#ifdef GL_ES\n"
"precision mediump float;\n"
"#endif\n"
"\n"
"uniform mat4 u_modelViewProjectionMatrix;\n"
"uniform mat4 u_modelMatrix;\n"
"uniform mat4 u_viewMatrix;\n"
"uniform mat4 u_projectionMatrix;\n"
"uniform mat4 u_normalMatrix;\n"
"\n"
"uniform float u_time;\n"
"uniform vec2 u_mouse;\n"
"uniform vec2 u_resolution;\n"
"\n"
"uniform vec3 u_eye3d;\n"
"uniform vec3 u_centre3d;\n"
"uniform vec3 u_up3d;\n"
"uniform mat3 u_view2d;\n"
"\n"
"#define iResolution vec3(u_resolution, 1.0)\n"
"\n";

    for (unsigned int i = 0; i < m_attribs.size(); i++) {
        int size = m_attribs[i].size;
        if (m_positionAttribIndex == int(i)) {
            size = 4;
        }
        rta += "in vec" + toString(size) + " a_" + m_attribs[i].name + ";\n";
        rta += "out vec" + toString(size) + " v_" + m_attribs[i].name + ";\n";
    }

    rta += "\n"
           "// * `eye` is the position of the camera.\n"
           "// * `centre` is the position to look towards.\n"
           "// * `up` is the 'up' direction.\n"
           "// * returns a 4x4 column major matrix.\n"
           "mat4 look_at(vec3 eye, vec3 centre, vec3 up)\n"
           "{\n"
           "    mat4 camMat;\n"
           "    vec3 ww = normalize(eye - centre);\n"
           "    vec3 uu = normalize(cross(up, ww));\n"
           "    vec3 vv = normalize(cross(ww, uu));\n"
           "    float du = dot(uu, -eye);\n"
           "    float dv = dot(vv, -eye);\n"
           "    float dw = dot(ww, -eye);\n"
           "    camMat[0][0] = uu.x;\n"
           "    camMat[0][1] = vv.x;\n"
           "    camMat[0][2] = ww.x;\n"
           "    camMat[0][3] = 0.0;\n"
           "    camMat[1][0] = uu.y;\n"
           "    camMat[1][1] = vv.y;\n"
           "    camMat[1][2] = ww.y;\n"
           "    camMat[1][3] = 0.0;\n"
           "    camMat[2][0] = uu.z;\n"
           "    camMat[2][1] = vv.z;\n"
           "    camMat[2][2] = ww.z;\n"
           "    camMat[2][3] = 0.0;\n"
           "    camMat[3][0] = du;\n"
           "    camMat[3][1] = dv;\n"
           "    camMat[3][2] = dw;\n"
           "    camMat[3][3] = 1.0;\n"
           "    return camMat;\n"
           "}\n";

    rta += "\n"
           "// * returns a 4x4 column major matrix.\n"
           "mat4 proj(float left, float right, float top, float bottom,\n"
           "          float n, float f)\n"
           "{\n"
           "    mat4 projMat;\n"
           "    projMat[0][0] = 2.0 * n / (right - left);\n"
           "    projMat[0][1] = 0.0;\n"
           "    projMat[0][2] = 0.0;\n"
           "    projMat[0][3] = 0.0;\n"
           "    projMat[1][0] = 0.0;\n"
           "    projMat[1][1] = 2.0 * n / (top - bottom);\n"
           "    projMat[1][2] = 0.0;\n"
           "    projMat[1][3] = 0.0;\n"
           "    projMat[2][0] = (right + left) / (right - left);\n"
           "    projMat[2][1] = (top + bottom) / (top - bottom);\n"
           "    projMat[2][2] = -(f + n) / (f - n);\n"
           "    projMat[2][3] = -1.0;\n"
           "    projMat[3][0] = 0.0;\n"
           "    projMat[3][1] = 0.0;\n"
           "    projMat[3][2] = - 2.0 * f * n / (f - n);\n"
           "    projMat[3][3] = 0.0;\n"
           "    return projMat;\n"
           "}\n";

    rta += "\n"
"void main(void) {\n"
"\n";

    for (unsigned int i = 0; i < m_attribs.size(); i++) {
        rta += "    v_" + m_attribs[i].name + " = a_" + m_attribs[i].name + ";\n";
    }

    rta += bbox;

    rta += "\n"
           "  float ar = iResolution.y/iResolution.x;\n"
           "  #ifdef BBOX_3D\n"
           "  const vec3 origin = (bbox_min + bbox_max) / 2.0;\n"
           "  const vec3 radius = (bbox_max - bbox_min) / 2.0;\n"
           "  float r = max(radius.x, max(radius.y, radius.z)) / 1.3;\n"
           "  vec3 eye = vec3(u_eye3d.x, -u_eye3d.z, u_eye3d.y)*r + origin;\n"
           "  vec3 centre = vec3(u_centre3d.x, -u_centre3d.z, u_centre3d.y)*r + origin;\n"
           "  vec3 up = vec3(u_up3d.x, -u_up3d.z, u_up3d.y);\n"
           "  mat4 camera = look_at(eye, centre, up);\n"
           "  mat4 projMat = proj(-1.0 / ar, 1.0 / ar, 1.0, -1.0, 2.5, 100);\n"
           "  #endif\n"
           "  #ifdef BBOX_2D\n"
           "  vec2 size = bbox.zw - bbox.xy;\n"
           "  vec2 origin2 = bbox.xy;\n"
           "  vec2 scale2 = size / iResolution.xy;\n"
           "  vec2 u_view2d_off = - (bbox.zw - bbox.xy) / 2.0;\n"
           "  vec2 offset = (bbox.zw + bbox.xy) / 2.0;\n"
           "  float scale;\n"
           "  float xscale = 1.0/u_view2d[0][0];\n"
           "  float yscale = 1.0/u_view2d[1][1];\n"
           "  if (scale2.x > scale2.y) {\n"
           "      scale = scale2.x;\n"
           "      u_view2d_off.y -= (iResolution.y*scale - size.y)/2.0;\n"
           "      yscale *= scale2.y/scale2.x / (scale2.y * iResolution.y / 2.0);\n"
           "      xscale /= scale2.x * iResolution.x / 2.0;\n"
           "  } else {\n"
           "      scale = scale2.y;\n"
           "      u_view2d_off.x -= (iResolution.x*scale - size.x)/2.0;\n"
           "      xscale *= scale2.x/scale2.y / (scale2.x * iResolution.x / 2.0);\n"
           "      yscale /= scale2.y * iResolution.y / 2.0;\n"
           "  }\n"
           "  vec2 xy = (u_view2d * vec3(0.0,0.0,1.0)).xy;\n"
           "  mat4 camera;\n"
           "  camera[0][0]=xscale; camera[1][0]=0.0; camera[2][0]=0.0; camera[3][0]=-(scale * xscale) * u_view2d[2][0] - (1.0-u_view2d[0][0]) * xscale * u_view2d_off.x - xscale * offset.x;\n"
           "  camera[0][1]=0.0; camera[1][1]=yscale; camera[2][1]=0.0; camera[3][1]=-(scale * yscale) * u_view2d[2][1] - (1.0-u_view2d[1][1])* yscale * u_view2d_off.y - yscale * offset.y;\n"
           "  camera[0][2]=0.0;            camera[1][2]=0.0;            camera[2][2]=1.0; camera[3][2]=0.0;\n"
           "  camera[0][3]=0.0;            camera[1][3]=0.0;            camera[2][3]=0.0; camera[3][3]=1.0;\n"
           "  mat4 projMat;\n"
           "  projMat[0][0]=1.0; projMat[1][0]=0.0; projMat[2][0]=0.0; projMat[3][0]=0.0;\n"
           "  projMat[0][1]=0.0; projMat[1][1]=1.0; projMat[2][1]=0.0; projMat[3][1]=0.0;\n"
           "  projMat[0][2]=0.0; projMat[1][2]=0.0; projMat[2][2]=1.0; projMat[3][2]=0.0;\n"
           "  projMat[0][3]=0.0; projMat[1][3]=0.0; projMat[2][3]=0.0; projMat[3][3]=1.0;\n"
           "  #endif\n"
           "\n";

    if (m_positionAttribIndex != -1 && m_positionAttribIndex < int(m_attribs.size())) {
        rta += "    gl_Position = projMat * camera * v_" + m_attribs[m_positionAttribIndex].name + ";\n";
    }

    rta +=  "}\n";

    return rta;
}

#endif

std::string VertexLayout::getDefaultVertShader() {
    std::string rta =
"#ifdef GL_ES\n"
"precision mediump float;\n"
"#endif\n"
"\n"
"uniform mat4 u_modelViewProjectionMatrix;\n"
"uniform mat4 u_modelMatrix;\n"
"uniform mat4 u_viewMatrix;\n"
"uniform mat4 u_projectionMatrix;\n"
"uniform mat4 u_normalMatrix;\n"
"\n"
"uniform float u_time;\n"
"uniform vec2 u_mouse;\n"
"uniform vec2 u_resolution;\n"
"\n";

    for (unsigned int i = 0; i < m_attribs.size(); i++) {
        int size = m_attribs[i].size;
        if (m_positionAttribIndex == int(i)) {
            size = 4;
        }
        rta += "in vec" + toString(size) + " a_" + m_attribs[i].name + ";\n";
        rta += "out vec" + toString(size) + " v_" + m_attribs[i].name + ";\n";
    }

    rta += "\n"
"void main(void) {\n"
"\n";

    for (unsigned int i = 0; i < m_attribs.size(); i++) {
        rta += "    v_" + m_attribs[i].name + " = a_" + m_attribs[i].name + ";\n";
    }

    if (m_positionAttribIndex != -1 && m_positionAttribIndex < int(m_attribs.size())) {
        rta += "    gl_Position = u_modelViewProjectionMatrix * v_" + m_attribs[m_positionAttribIndex].name + ";\n";
    }

    rta +=  "}\n";

    return rta;
}

std::string VertexLayout::getDefaultFragShader() {
    std::string rta =
"#ifdef GL_ES\n"
"precision mediump float;\n"
"#endif\n"
"\n"
"uniform mat4 u_modelViewProjectionMatrix;\n"
"uniform mat4 u_modelMatrix;\n"
"uniform mat4 u_viewMatrix;\n"
"uniform mat4 u_projectionMatrix;\n"
"uniform mat4 u_normalMatrix;\n"
"\n"
"uniform float u_time;\n"
"uniform vec2 u_mouse;\n"
"uniform vec2 u_resolution;\n"
"\n";

    for (unsigned int i = 0; i < m_attribs.size(); i++) {
        int size = m_attribs[i].size;
        if (m_positionAttribIndex == int(i)) {
            size = 4;
        }
        rta += "in vec" + toString(size) + " v_" + m_attribs[i].name + ";\n";
    }

    rta += "\n"
"void main(void) {\n"
"\n";

    if (m_colorAttribIndex != -1) {
        rta += "    gl_FragColor = v_" + m_attribs[m_colorAttribIndex].name + ";\n";
    }
    else if ( m_texCoordAttribIndex != -1 ){
        rta += "    gl_FragColor = vec4(vec3(v_" + m_attribs[m_texCoordAttribIndex].name + ",1.0),1.0);\n";
    }
    else if ( m_normalAttribIndex != -1 ){
        rta += "    gl_FragColor = vec4(0.5+v_" + m_attribs[m_normalAttribIndex].name + "*0.5,1.0);\n";
    }
    else {
        rta += "    gl_FragColor = vec4(1.0);\n";
    }

    rta +=  "}\n";

    return rta;
}
