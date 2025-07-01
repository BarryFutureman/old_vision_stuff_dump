import bpy


class SimpleRenderPanel(bpy.types.Panel):
    bl_label = "SD Draw"
    bl_idname = "PT_SD_Draw_Panel"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = 'SD Draw'

    def draw(self, context):
        layout = self.layout

        # Add a button to the panel
        layout.operator("render.simple_render", text="Render Scene")


class SimpleRenderOperator(bpy.types.Operator):
    bl_idname = "render.simple_render"
    bl_label = "Render Scene"

    def execute(self, context):
        # Render the scene
        bpy.ops.render.render(write_still=True)
        return {'FINISHED'}


def register():
    bpy.utils.register_class(SimpleRenderPanel)
    bpy.utils.register_class(SimpleRenderOperator)


def unregister():
    bpy.utils.unregister_class(SimpleRenderPanel)
    bpy.utils.unregister_class(SimpleRenderOperator)


if __name__ == "__main__":
    register()
