package net.minecraft.client.gui.screen.inventory;

import com.google.gson.JsonObject;
import com.google.gson.JsonArray;
import java.lang.reflect.Field;
import java.util.List;
import net.minecraft.client.Minecraft;
import net.minecraft.client.gui.recipebook.*;
import net.minecraft.client.gui.widget.Widget;
import net.minecraft.inventory.container.Container;
import net.minecraft.inventory.container.Slot;
import net.minecraft.item.Item;
import net.minecraft.item.ItemStack;
import net.minecraft.entity.player.ServerPlayerEntity;
import net.minecraft.util.ResourceLocation;
import net.minecraft.util.registry.Registry;
import net.minecraft.util.math.*;
import net.minecraft.util.math.vector.Vector3d;
import net.minecraft.util.Direction;
import net.minecraft.block.Blocks;

/** Read-only local workshop telemetry. No placement, recipe, inventory, or unlock writes. */
public final class WorkshopObservation {
    private static final WoodTracker legacyWood=new WoodTracker();
    private static final WoodTracker interactionWood=new WoodTracker();
    private static final class WoodTracker {
        Object woodWorld;
        BlockPos woodTarget;
        long woodMissingSince=-1;
    }
    private static boolean logBlock(ServerPlayerEntity sp, BlockPos p) {
        String id=Registry.BLOCK.getKey(sp.getServerWorld().getBlockState(p).getBlock()).getPath();
        return id.equals("oak_log") || id.equals("birch_log") || id.equals("spruce_log") ||
            id.equals("jungle_log") || id.equals("acacia_log") || id.equals("dark_oak_log");
    }
    private static final boolean cameraWoodEnabled=Boolean.getBoolean("minecraftRL.cameraWood");
    // Read-only visible-surface rays inside the fixed 70-degree vertical FOV.
    // Every ray stops at its FIRST opaque outline; never enumerate logs behind walls.
    private static BlockPos cameraWood(JsonObject j,ServerPlayerEntity sp,Vector3d eye) {
        BlockPos chosen=null;double best=Double.POSITIVE_INFINITY;
        JsonObject evidence=null;
        for(int yaw=-40;yaw<=40;yaw+=5)for(int pitch=-20;pitch<=20;pitch+=5) {
            double cost=yaw*yaw+pitch*pitch;
            if(cost>=best)continue;
            Vector3d look=Vector3d.fromPitchYaw(MathHelper.clamp(sp.rotationPitch+pitch,-89F,89F),sp.rotationYaw+yaw);
            BlockRayTraceResult ray=sp.getServerWorld().rayTraceBlocks(new RayTraceContext(eye,
                eye.add(look.scale(16)),RayTraceContext.BlockMode.OUTLINE,RayTraceContext.FluidMode.NONE,sp));
            if(ray.getType()!=RayTraceResult.Type.BLOCK || !logBlock(sp,ray.getPos()))continue;
            chosen=ray.getPos().toImmutable();best=cost;evidence=new JsonObject();
            evidence.addProperty("source","privileged_camera_surface_rays_v1");
            number(evidence,"tick",sp.getServer().getTickCounter());
            number(evidence,"yaw_offset",yaw);number(evidence,"pitch_offset",pitch);
            number(evidence,"yaw",sp.rotationYaw);number(evidence,"pitch",sp.rotationPitch);
            number(evidence,"range",16);
            number(evidence,"eye_x",eye.x);number(evidence,"eye_y",eye.y);number(evidence,"eye_z",eye.z);
            number(evidence,"hit_x",ray.getHitVec().x);number(evidence,"hit_y",ray.getHitVec().y);number(evidence,"hit_z",ray.getHitVec().z);
            number(evidence,"block_x",chosen.getX());number(evidence,"block_y",chosen.getY());number(evidence,"block_z",chosen.getZ());
        }
        if(evidence!=null)j.add("wood_camera_acquisition",evidence);
        return chosen;
    }
    private static void wood(JsonObject j,ServerPlayerEntity sp,WoodTracker state,String prefix,BlockPos mining) {
        number(j,prefix+"kind",0);number(j,prefix+"distance",0);number(j,prefix+"reachable",0);
        for(String axis:new String[]{"x","y","z"})number(j,prefix+"target_"+axis,0);
        if(sp==null)return;
        if(state.woodWorld!=sp.getServerWorld()) {state.woodWorld=sp.getServerWorld();state.woodTarget=null;state.woodMissingSince=-1;}
        BlockPos woodTarget=state.woodTarget;long woodMissingSince=state.woodMissingSince;
        Vector3d eye=sp.getEyePosition(1);long tick=sp.getServerWorld().getGameTime();
        if(woodTarget!=null) {
            Vector3d center=Vector3d.copyCentered(woodTarget);
            if(eye.squareDistanceTo(center)>32*32) {woodTarget=null;woodMissingSince=-1;}
            else if(!logBlock(sp,woodTarget)) {
                if(woodMissingSince<0)woodMissingSince=tick;
                if(tick-woodMissingSince>=40) {woodTarget=null;woodMissingSince=-1;}
            } else woodMissingSince=-1;
        }
        // Only an accepted, ongoing server log interaction may replace a live lock.
        // Looking away does not discard the goal. Actual new mining supersedes pickup intent.
        if(mining!=null) {woodTarget=mining.toImmutable();woodMissingSince=-1;}
        if(woodTarget==null) {
            BlockRayTraceResult ray=sp.getServerWorld().rayTraceBlocks(new RayTraceContext(eye,
                eye.add(sp.getLook(1).scale(16)),RayTraceContext.BlockMode.OUTLINE,RayTraceContext.FluidMode.NONE,sp));
            if(ray.getType()==RayTraceResult.Type.BLOCK && logBlock(sp,ray.getPos()))woodTarget=ray.getPos().toImmutable();
            if(woodTarget==null && state==interactionWood && cameraWoodEnabled)woodTarget=cameraWood(j,sp,eye);
        }
        state.woodTarget=woodTarget;state.woodMissingSince=woodMissingSince;
        if(woodTarget==null)return;
        Vector3d center=Vector3d.copyCentered(woodTarget);
        number(j,prefix+"kind",woodMissingSince<0?1:2);number(j,prefix+"distance",eye.distanceTo(center));
        number(j,prefix+"target_x",woodTarget.getX());number(j,prefix+"target_y",woodTarget.getY());number(j,prefix+"target_z",woodTarget.getZ());
        BlockRayTraceResult ray=sp.getServerWorld().rayTraceBlocks(new RayTraceContext(eye,center,
            RayTraceContext.BlockMode.OUTLINE,RayTraceContext.FluidMode.NONE,sp));
        number(j,prefix+"reachable",mining!=null || (woodMissingSince<0 && eye.distanceTo(center)<=4.0 &&
            ray.getType()==RayTraceResult.Type.BLOCK && ray.getPos().equals(woodTarget))?1:0);
    }
    private static volatile JsonObject snapshot;
    public static void capture() {snapshot=read();}
    public static JsonObject snapshot() {
        if(snapshot==null)throw new IllegalStateException("workshop snapshot not ready");
        return snapshot;
    }
    private static Object field(Object object, String name) throws ReflectiveOperationException {
        Field f=object.getClass().getDeclaredField(name);f.setAccessible(true);return f.get(object);
    }
    private static String name(ItemStack stack) { return Registry.ITEM.getKey(stack.getItem()).getPath(); }
    private static int index(Slot slot) {
        try {Field f=Slot.class.getDeclaredField("slotIndex");f.setAccessible(true);return f.getInt(slot);}
        catch(ReflectiveOperationException e) {throw new IllegalStateException(e);}
    }
    private static void number(JsonObject j,String key,double value) {j.addProperty(key,value);}
    // Conservative, bounded, read-only walking sensor. No resource search or world-file access.
    // Only full support + two empty, dry body cells are advertised; stairs/jumps remain unknown.
    private static void localNavigation(JsonObject j,ServerPlayerEntity sp) {
        if(sp==null)return;
        int r=6,x=(int)Math.floor(sp.getPosX()),y=(int)Math.floor(sp.getPosY()+.01),z=(int)Math.floor(sp.getPosZ());
        // While jumping, scan the real support below the player, not an empty airborne plane.
        // At most two levels; unsupported drops are still not advertised as walking routes.
        for(int below=0;below<=2;below++) {
            BlockPos floor=new BlockPos(x,y-below-1,z);
            if(sp.getServerWorld().isBlockLoaded(floor) &&
               sp.getServerWorld().getBlockState(floor).isSolidSide(sp.getServerWorld(),floor,Direction.UP)) {
                y-=below;break;
            }
        }
        JsonObject nav=new JsonObject();JsonArray cells=new JsonArray();JsonArray access=new JsonArray();
        number(nav,"radius",r);number(nav,"x",x);number(nav,"y",y);number(nav,"z",z);
        number(nav,"tick",sp.getServer().getTickCounter());
        nav.addProperty("source","privileged_local_passability_v1");
        BlockPos target=interactionWood.woodTarget;
        for(int dz=-r;dz<=r;dz++)for(int dx=-r;dx<=r;dx++) {
            BlockPos p=new BlockPos(x+dx,y,z+dz);boolean open=false,ready=false;
            if(sp.getServerWorld().isBlockLoaded(p)) {
                open=sp.getServerWorld().isAirBlock(p) && sp.getServerWorld().isAirBlock(p.up()) &&
                    sp.getServerWorld().getBlockState(p.down()).isSolidSide(sp.getServerWorld(),p.down(),Direction.UP) &&
                    sp.getServerWorld().getFluidState(p).isEmpty() && sp.getServerWorld().getFluidState(p.up()).isEmpty();
                if(open && target!=null && logBlock(sp,target)) {
                    Vector3d eye=new Vector3d(p.getX()+.5,y+1.62,p.getZ()+.5),center=Vector3d.copyCentered(target);
                    if(eye.distanceTo(center)<=3.5) {
                        BlockRayTraceResult ray=sp.getServerWorld().rayTraceBlocks(new RayTraceContext(eye,center,
                            RayTraceContext.BlockMode.OUTLINE,RayTraceContext.FluidMode.NONE,sp));
                        ready=ray.getType()==RayTraceResult.Type.BLOCK && ray.getPos().equals(target);
                    }
                }
            }
            cells.add(open?1:0);access.add(ready?1:0);
        }
        nav.add("walkable",cells);nav.add("wood_access",access);j.add("local_navigation",nav);
    }
    public static JsonObject read() {
        Minecraft mc=Minecraft.getInstance();JsonObject j=new JsonObject();
        number(j,"wood_camera_rays_enabled",cameraWoodEnabled?1:0);
        ServerPlayerEntity sp=mc.player==null || mc.getIntegratedServer()==null ? null :
            mc.getIntegratedServer().getPlayerList().getPlayerByUUID(mc.player.getUniqueID());
        boolean inv=mc.currentScreen instanceof InventoryScreen,table=mc.currentScreen instanceof CraftingScreen;
        ContainerScreen<?> screen=inv || table ? (ContainerScreen<?>)mc.currentScreen : null;
        RecipeBookGui book=inv ? ((InventoryScreen)screen).getRecipeGui() : table ? ((CraftingScreen)screen).getRecipeGui() : null;
        number(j,"screen",inv?1:table?2:mc.currentScreen==null?0:3);number(j,"width",inv?2:table?3:0);
        number(j,"error",0);number(j,"book_open",book!=null && book.isVisible()?1:0);
        double scale=mc.mouseHelper.getScaleFactor();
        number(j,"mouse_x",mc.mouseHelper.getMouseX()/scale);number(j,"mouse_y",mc.mouseHelper.getMouseY()/scale);
        number(j,"book_x",screen==null?-1:screen.guiLeft+(inv?114:15));
        number(j,"book_y",screen==null?-1:screen.height/2+(inv?-13:-40));
        ItemStack cursor=mc.player==null?ItemStack.EMPTY:mc.player.inventory.getItemStack();
        j.addProperty("cursor_item",name(cursor));number(j,"cursor_count",cursor.getCount());
        number(j,"server_available",sp==null?0:1);number(j,"server_cursor",sp==null?-1:sp.inventory.getItemStack().getCount());
        number(j,"server_health",sp==null?-1:sp.getHealth());number(j,"server_alive",sp!=null && sp.isAlive()?1:0);
        int heldKeys=0;
        for(net.minecraft.client.settings.KeyBinding key:mc.gameSettings.keyBindings)if(key.isKeyDown())heldKeys++;
        number(j,"held_key_count",heldKeys);
        number(j,"server_tick",mc.getIntegratedServer()==null?-1:mc.getIntegratedServer().getTickCounter());
        number(j,"render_ready",mc.world!=null && mc.worldRenderer.hasNoChunkUpdates()?1:0);
        number(j,"world_time",sp==null?-1:sp.getServerWorld().getGameTime());
        number(j,"server_x",sp==null?0:sp.getPosX());number(j,"server_y",sp==null?0:sp.getPosY());number(j,"server_z",sp==null?0:sp.getPosZ());
        JsonObject inventory=new JsonObject();
        for (ResourceLocation key:Registry.ITEM.keySet())number(inventory,key.getPath(),0);
        if (sp!=null)for(ItemStack stack:sp.inventory.mainInventory)
            number(inventory,name(stack),inventory.get(name(stack)).getAsInt()+stack.getCount());
        j.add("server_inventory",inventory);
        BlockPos mining=sp!=null && sp.isAlive()?sp.interactionManager.minecraftRLMiningTarget():null;
        if(mining!=null && !logBlock(sp,mining))mining=null;
        number(j,"wood_server_mining",mining==null?0:1);
        for(String axis:new String[]{"x","y","z"})number(j,"wood_mining_"+axis,0);
        if(mining!=null) {
            number(j,"wood_mining_x",mining.getX());number(j,"wood_mining_y",mining.getY());number(j,"wood_mining_z",mining.getZ());
        }
        wood(j,sp,legacyWood,"wood_",null);
        wood(j,sp,interactionWood,"wood_v3_",mining);
        localNavigation(j,sp);
        JsonArray slots=new JsonArray();Container container=screen==null?null:screen.getContainer();
        for(int i=0;i<46;i++) {
            JsonObject s=new JsonObject();Slot slot=container!=null && i<container.inventorySlots.size()?container.inventorySlots.get(i):null;
            ItemStack stack=slot==null?ItemStack.EMPTY:slot.getStack();s.addProperty("item",name(stack));number(s,"count",stack.getCount());
            number(s,"x",slot==null?-1:screen.guiLeft+slot.xPos+8);number(s,"y",slot==null?-1:screen.guiTop+slot.yPos+8);
            number(s,"inventory_index",slot!=null && slot.inventory==mc.player.inventory?index(slot):-1);
            slots.add(s);
        }
        j.add("slots",slots);
        String[] ids={"oak_planks","birch_planks","spruce_planks","jungle_planks","acacia_planks","dark_oak_planks",
            "stick","crafting_table","wooden_pickaxe","wooden_axe","oak_slab"};
        JsonObject recipes=new JsonObject();
        for(String id:ids) {
            JsonObject r=new JsonObject();ResourceLocation key=new ResourceLocation("minecraft",id);
            number(r,"known",mc.player!=null && mc.player.getRecipeBook().isUnlocked(key)?1:0);
            number(r,"server_known",sp!=null && sp.getRecipeBook().isUnlocked(key)?1:0);
            number(r,"x",-1);number(r,"y",-1);number(r,"single",0);number(r,"craftable",0);recipes.add(id,r);
        }
        j.add("recipes",recipes);
        for(String key:new String[]{"next_x","next_y","previous_x","previous_y","search_x","search_y","page","pages"})number(j,key,-1);
        number(j,"search_tab",0);
        if(book!=null && book.isVisible())try {
            RecipeBookPage page=(RecipeBookPage)field(book,"recipeBookPage");
            for(RecipeWidget button:(List<RecipeWidget>)field(page,"buttons")) {
                if(!button.visible)continue;String id=button.getRecipe().getId().getPath();
                if(!recipes.has(id))continue;JsonObject r=recipes.getAsJsonObject(id);
                number(r,"x",button.x+12);number(r,"y",button.y+12);number(r,"single",button.isOnlyOption()?1:0);
                number(r,"craftable",button.getList().isCraftable(button.getRecipe())?1:0);
            }
            number(j,"page",(Integer)field(page,"currentPage"));number(j,"pages",(Integer)field(page,"totalPages"));
            for(String direction:new String[]{"next","previous"}) {
                Widget button=(Widget)field(page,direction.equals("next")?"forwardButton":"backButton");
                if(button.visible) {number(j,direction+"_x",button.x+6);number(j,direction+"_y",button.y+8);}
            }
            List<RecipeTabToggleWidget> tabs=(List<RecipeTabToggleWidget>)field(book,"recipeTabs");
            RecipeTabToggleWidget tab=tabs.get(0);
            number(j,"search_x",tab.x+12);number(j,"search_y",tab.y+12);
            number(j,"search_tab",field(book,"currentTab")==tab?1:0);
        } catch(ReflectiveOperationException | RuntimeException error) {number(j,"error",1);}
        number(j,"yaw",mc.player==null?0:mc.player.rotationYaw);number(j,"pitch",mc.player==null?0:mc.player.rotationPitch);
        Vector3d eye=mc.player==null?Vector3d.ZERO:mc.player.getEyePosition(1);
        number(j,"eye_x",eye.x);number(j,"eye_y",eye.y);number(j,"eye_z",eye.z);
        j.addProperty("selected_item",mc.player==null?"air":name(mc.player.getHeldItemMainhand()));
        for(String key:new String[]{"hit_x","hit_y","hit_z","hit_face","candidate_x","candidate_y","candidate_z",
                                    "table_x","table_y","table_z"})number(j,key,-999);
        number(j,"hit_table",0);number(j,"table_count",0);number(j,"candidate",0);
        if(mc.objectMouseOver instanceof BlockRayTraceResult && mc.objectMouseOver.getType()==RayTraceResult.Type.BLOCK) {
            BlockRayTraceResult hit=(BlockRayTraceResult)mc.objectMouseOver;BlockPos p=hit.getPos();
            number(j,"hit_x",p.getX());number(j,"hit_y",p.getY());number(j,"hit_z",p.getZ());number(j,"hit_face",hit.getFace().ordinal());
            number(j,"hit_table",mc.world.getBlockState(p).getBlock()==Blocks.CRAFTING_TABLE?1:0);
        }
        if(mc.player!=null && mc.world!=null && sp!=null) {
            BlockPos origin=mc.player.getPosition();double best=Double.POSITIVE_INFINITY,tableDistance=Double.POSITIVE_INFINITY;int tables=0;
            for(int dx=-4;dx<=4;dx++)for(int dz=-4;dz<=4;dz++)for(int dy=-3;dy<=2;dy++) {
                BlockPos p=origin.add(dx,dy,dz);Vector3d top=new Vector3d(p.getX()+.5,p.getY()+.999,p.getZ()+.5);
                double distance=eye.squareDistanceTo(top);
                if(sp.getServerWorld().getBlockState(p).getBlock()==Blocks.CRAFTING_TABLE) {
                    tables++;if(distance<tableDistance) {tableDistance=distance;number(j,"table_x",p.getX());number(j,"table_y",p.getY());number(j,"table_z",p.getZ());}
                }
                if(distance>3.7*3.7 || distance>=best || !mc.world.isAirBlock(p.up()) || !sp.getServerWorld().isAirBlock(p.up()) ||
                    mc.player.getBoundingBox().intersects(new AxisAlignedBB(p.up())))continue;
                if(!mc.world.getBlockState(p).isSolidSide(mc.world,p,Direction.UP))continue;
                // Avoid opening an interactive support block instead of placing.
                if(mc.world.getBlockState(p).getBlock()!=Blocks.GRASS_BLOCK && mc.world.getBlockState(p).getBlock()!=Blocks.DIRT &&
                   mc.world.getBlockState(p).getBlock()!=Blocks.STONE && mc.world.getBlockState(p).getBlock()!=Blocks.COBBLESTONE)continue;
                BlockRayTraceResult ray=mc.world.rayTraceBlocks(new RayTraceContext(eye,top,RayTraceContext.BlockMode.OUTLINE,RayTraceContext.FluidMode.NONE,mc.player));
                if(ray.getType()!=RayTraceResult.Type.BLOCK || !ray.getPos().equals(p) || ray.getFace()!=Direction.UP)continue;
                best=distance;number(j,"candidate",1);number(j,"candidate_x",p.getX());number(j,"candidate_y",p.getY()+1);number(j,"candidate_z",p.getZ());
            }
            number(j,"table_count",tables);
        }
        return j;
    }
}
