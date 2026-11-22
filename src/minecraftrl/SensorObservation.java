package net.minecraft.client.gui.screen.inventory;

import com.google.gson.JsonArray;
import com.google.gson.JsonObject;
import net.minecraft.block.Block;
import net.minecraft.block.BlockState;
import net.minecraft.client.Minecraft;
import net.minecraft.client.entity.player.ClientPlayerEntity;
import net.minecraft.fluid.FluidState;
import net.minecraft.item.ItemStack;
import net.minecraft.tags.FluidTags;
import net.minecraft.util.ResourceLocation;
import net.minecraft.util.math.BlockPos;
import net.minecraft.util.math.BlockRayTraceResult;
import net.minecraft.util.math.RayTraceContext;
import net.minecraft.util.math.RayTraceResult;
import net.minecraft.util.math.vector.Vector3d;
import net.minecraft.util.registry.Registry;

/** Read-only client sensor telemetry (sensor-tier design 2026-11-19 §3.1): the 27 blocks around the feet, 136 view rays
 *  (a 15x9 grid over the field of view plus the crosshair) with the first fluid nearer than the solid hit, and the held
 *  tool's wear. Reads only the client world and the client player, like a body-mounted sensor. Opt-in: only jars built
 *  with the sensor monitor carry this class. */
public final class SensorObservation {
    static final int COLS=15,ROWS=9,RAYS=COLS*ROWS+1;
    static final double RANGE=8.;
    private static volatile JsonObject snapshot;
    public static void capture() {snapshot=read();}
    public static JsonObject snapshot() {
        if(snapshot==null)throw new IllegalStateException("sensor snapshot not ready");
        return snapshot;
    }
    private static void number(JsonObject j,String key,double value) {j.addProperty(key,value);}
    // Item vocabulary as in DemoObservation.block(), plus fire (hazard guard); every other non-item block is "other".
    private static String kind(BlockState state) {
        Block b=state.getBlock();String id=Registry.BLOCK.getKey(b).getPath();
        if(id.equals("fire") || id.equals("soul_fire"))return id;
        return Registry.ITEM.containsKey(new ResourceLocation("minecraft",id))?id:"other";
    }
    private static String fluid(FluidState f) {
        if(f==null || f.isEmpty())return "none";
        return f.isTagged(FluidTags.LAVA)?"lava":f.isTagged(FluidTags.WATER)?"water":"none";
    }
    private static Vector3d rotation(double yaw,double pitch) {
        double y=Math.toRadians(yaw),p=Math.toRadians(pitch);
        return new Vector3d(-Math.sin(y)*Math.cos(p),-Math.sin(p),Math.cos(y)*Math.cos(p));
    }
    public static JsonObject read() {
        Minecraft mc=Minecraft.getInstance();JsonObject j=new JsonObject();
        ClientPlayerEntity player=mc.player;
        JsonArray kinds=new JsonArray(),dist=new JsonArray(),rx=new JsonArray(),ry=new JsonArray(),rz=new JsonArray(),
            dx=new JsonArray(),dy=new JsonArray(),dz=new JsonArray(),fluids=new JsonArray(),fluidDist=new JsonArray(),
            voxels=new JsonArray(),voxelFluids=new JsonArray();
        boolean available=player!=null && mc.world!=null;
        number(j,"sensor_available",available?1:0);
        number(j,"sensor_cols",COLS);number(j,"sensor_rows",ROWS);number(j,"sensor_range",RANGE);
        double vfov=mc.gameSettings==null?70.:mc.gameSettings.fov;
        double aspect=16./9.;
        if(mc.getMainWindow()!=null && mc.getMainWindow().getFramebufferHeight()>0)
            aspect=(double)mc.getMainWindow().getFramebufferWidth()/mc.getMainWindow().getFramebufferHeight();
        double hfov=Math.toDegrees(2*Math.atan(Math.tan(Math.toRadians(vfov)/2)*aspect));
        number(j,"sensor_vfov",vfov);number(j,"sensor_hfov",hfov);
        if(!available) {
            number(j,"sensor_client_tick",-1);
            for(String key:new String[]{"sensor_eye_x","sensor_eye_y","sensor_eye_z","sensor_yaw","sensor_pitch",
                "sensor_foot_x","sensor_foot_y","sensor_foot_z","sensor_tool_damage","sensor_tool_max"})number(j,key,0);
            j.addProperty("sensor_tool_item","air");
        } else {
            number(j,"sensor_client_tick",mc.world.getGameTime());
            Vector3d eye=player.getEyePosition(1);
            double yaw=player.rotationYaw,pitch=player.rotationPitch;
            number(j,"sensor_eye_x",eye.x);number(j,"sensor_eye_y",eye.y);number(j,"sensor_eye_z",eye.z);
            number(j,"sensor_yaw",yaw);number(j,"sensor_pitch",pitch);
            Vector3d forward=rotation(yaw,pitch);
            double y=Math.toRadians(yaw);
            Vector3d right=new Vector3d(-Math.cos(y),0,-Math.sin(y));
            Vector3d up=right.crossProduct(forward).normalize();
            double th=Math.tan(Math.toRadians(hfov)/2),tv=Math.tan(Math.toRadians(vfov)/2);
            for(int i=0;i<RAYS;i++) {
                Vector3d dir;
                if(i<COLS*ROWS) {
                    int col=i%COLS,row=i/COLS;
                    double sx=2.*col/(COLS-1)-1,sy=1-2.*row/(ROWS-1);
                    dir=forward.add(right.scale(sx*th)).add(up.scale(sy*tv)).normalize();
                } else dir=forward;
                Vector3d end=eye.add(dir.scale(RANGE));
                BlockRayTraceResult solid=mc.world.rayTraceBlocks(new RayTraceContext(eye,end,
                    RayTraceContext.BlockMode.OUTLINE,RayTraceContext.FluidMode.NONE,player));
                BlockPos hit=null;double d=-1;
                if(i==COLS*ROWS) {     // the crosshair: the client's own pick (reach-limited), as demo_hit_block
                    if(mc.objectMouseOver instanceof BlockRayTraceResult && mc.objectMouseOver.getType()==RayTraceResult.Type.BLOCK) {
                        BlockRayTraceResult pick=(BlockRayTraceResult)mc.objectMouseOver;
                        hit=pick.getPos();d=pick.getHitVec().distanceTo(eye);
                    }
                } else if(solid!=null && solid.getType()==RayTraceResult.Type.BLOCK) {
                    hit=solid.getPos();d=solid.getHitVec().distanceTo(eye);
                }
                kinds.add(hit==null?"air":kind(mc.world.getBlockState(hit)));dist.add(d);
                rx.add(hit==null?-999:hit.getX());ry.add(hit==null?-999:hit.getY());rz.add(hit==null?-999:hit.getZ());
                dx.add(dir.x);dy.add(dir.y);dz.add(dir.z);
                BlockRayTraceResult wet=mc.world.rayTraceBlocks(new RayTraceContext(eye,end,
                    RayTraceContext.BlockMode.OUTLINE,RayTraceContext.FluidMode.ANY,player));
                String f="none";double fd=-1;
                if(wet!=null && wet.getType()==RayTraceResult.Type.BLOCK) {
                    double wd=wet.getHitVec().distanceTo(eye);
                    String wf=fluid(mc.world.getFluidState(wet.getPos()));
                    if(!wf.equals("none") && (d<0 || wd<d)) {f=wf;fd=wd;}
                }
                fluids.add(f);fluidDist.add(fd);
            }
            BlockPos foot=new BlockPos(player.getPosX(),player.getPosY(),player.getPosZ());
            number(j,"sensor_foot_x",foot.getX());number(j,"sensor_foot_y",foot.getY());number(j,"sensor_foot_z",foot.getZ());
            for(int oy=-1;oy<=1;oy++)for(int oz=-1;oz<=1;oz++)for(int ox=-1;ox<=1;ox++) {
                BlockPos p=foot.add(ox,oy,oz);
                voxels.add(kind(mc.world.getBlockState(p)));voxelFluids.add(fluid(mc.world.getFluidState(p)));
            }
            ItemStack held=player.getHeldItemMainhand();
            j.addProperty("sensor_tool_item",Registry.ITEM.getKey(held.getItem()).getPath());
            number(j,"sensor_tool_damage",held.isDamageable()?held.getDamage():0);
            number(j,"sensor_tool_max",held.isDamageable()?held.getMaxDamage():0);
        }
        j.add("sensor_ray_kind",kinds);j.add("sensor_ray_dist",dist);
        j.add("sensor_ray_x",rx);j.add("sensor_ray_y",ry);j.add("sensor_ray_z",rz);
        j.add("sensor_ray_dx",dx);j.add("sensor_ray_dy",dy);j.add("sensor_ray_dz",dz);
        j.add("sensor_ray_fluid",fluids);j.add("sensor_ray_fluid_dist",fluidDist);
        j.add("sensor_voxel_kind",voxels);j.add("sensor_voxel_fluid",voxelFluids);
        return j;
    }
}
