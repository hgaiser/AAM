uniform sampler2D input_tex;
uniform sampler2D mean_tex;
//uniform sampler2D filter_tex;

void main(void)
{
   vec4 texval1 = texture2D(input_tex,  vec2(gl_TexCoord[0]));
   vec4 texval2 = texture2D(mean_tex,   vec2(gl_TexCoord[1]));
   //vec4 texval3 = texture2D(filter_tex, vec2(gl_TexCoord[1]));
   
   gl_FragColor = ((0.5 * texval1) + 0.5) - (0.5 * texval2);
}

