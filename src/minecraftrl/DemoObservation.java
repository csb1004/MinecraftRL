package net.minecraft.client.gui.screen.inventory;

import com.google.gson.JsonObject;
import net.minecraft.block.Block;
import net.minecraft.client.Minecraft;
import net.minecraft.entity.player.ServerPlayerEntity;
import net.minecraft.server.management.PlayerInteractionManager;
import net.minecraft.util.ResourceLocation;
import net.minecraft.util.math.BlockPos;
import net.minecraft.util.math.BlockRayTraceResult;
import net.minecraft.util.math.RayTraceResult;
import net.minecraft.util.registry.Registry;

/** Read-only demo telemetry: crosshair block, server mining target of any block, every survival destroy. */
public final class DemoObservation {
    private static volatile JsonObject snapshot;
    public static void capture() {DemoFixture.tick();snapshot=read();}
    public static JsonObject snapshot() {
        if(snapshot==null)throw new IllegalStateException("demo snapshot not ready");
        return snapshot;
    }
    private static void number(JsonObject j,String key,double value) {j.addProperty(key,value);}
    // Block ids outside the item vocabulary (e.g. wall_torch, water) are reported as "other".
    private static String block(Block b) {
        String id=Registry.BLOCK.getKey(b).getPath();
        return Registry.ITEM.containsKey(new ResourceLocation("minecraft",id))?id:"other";
    }
    private static void position(JsonObject j,String prefix,BlockPos p) {
        number(j,prefix+"x",p==null?-999:p.getX());number(j,prefix+"y",p==null?-999:p.getY());number(j,prefix+"z",p==null?-999:p.getZ());
    }
    public static JsonObject read() {
        Minecraft mc=Minecraft.getInstance();JsonObject j=new JsonObject();
        ServerPlayerEntity sp=mc.player==null || mc.getIntegratedServer()==null ? null :
            mc.getIntegratedServer().getPlayerList().getPlayerByUUID(mc.player.getUniqueID());
        PlayerInteractionManager im=sp==null?null:sp.interactionManager;
        number(j,"demo_available",sp==null?0:1);
        number(j,"demo_destroys",im==null?0:im.minecraftRLDestroys);
        number(j,"demo_last_destroy_tick",im==null?-1:im.minecraftRLLastDestroyTick);
        j.addProperty("demo_last_destroyed_block",im==null?"air":im.minecraftRLLastDestroyed);
        j.addProperty("demo_last_destroy_tool",im==null?"air":im.minecraftRLLastDestroyTool);
        number(j,"demo_last_destroy_removed",im!=null && im.minecraftRLLastDestroyRemoved?1:0);
        // Outcome evidence only; the guard decides from aim, held item and stored knowledge.
        number(j,"demo_last_destroy_harvestable",im!=null && im.minecraftRLLastDestroyHarvestable?1:0);
        position(j,"demo_last_destroy_",im==null || im.minecraftRLDestroys==0?null:im.minecraftRLLastDestroy);
        BlockPos mining=sp!=null && sp.isAlive()?im.minecraftRLMiningTarget():null;
        j.addProperty("demo_mining_block",mining==null?"air":block(sp.getServerWorld().getBlockState(mining).getBlock()));
        position(j,"demo_mining_",mining);
        BlockPos hit=null;
        if(mc.world!=null && mc.objectMouseOver instanceof BlockRayTraceResult && mc.objectMouseOver.getType()==RayTraceResult.Type.BLOCK)
            hit=((BlockRayTraceResult)mc.objectMouseOver).getPos();
        j.addProperty("demo_hit_block",hit==null?"air":block(mc.world.getBlockState(hit).getBlock()));
        number(j,"demo_fixture_state",DemoFixture.state());
        number(j,"demo_fixture_blocks",DemoFixture.blocks());
        number(j,"demo_fixture_sha52",DemoFixture.sha52());
        if(DemoFixture.state()==2)number(j,"demo_fixture_origin_y",DemoFixture.originY());
        return j;
    }
}
