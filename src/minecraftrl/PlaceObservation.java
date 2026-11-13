package net.minecraft.client.gui.screen.inventory;

import com.google.gson.JsonObject;
import net.minecraft.client.Minecraft;
import net.minecraft.entity.player.ServerPlayerEntity;
import net.minecraft.server.management.PlayerInteractionManager;
import net.minecraft.util.math.BlockPos;

/** Read-only placement telemetry (block placing plan 2026-11-13 §1): every survival block a player placed on the server
 *  (the crosshair block and its face are already in the workshop telemetry). Opt-in: only jars built with the place monitor carry this class. */
public final class PlaceObservation {
    private static volatile JsonObject snapshot;
    public static void capture() {snapshot=read();}
    public static JsonObject snapshot() {
        if(snapshot==null)throw new IllegalStateException("place snapshot not ready");
        return snapshot;
    }
    private static void number(JsonObject j,String key,double value) {j.addProperty(key,value);}
    private static void position(JsonObject j,String prefix,BlockPos p) {
        number(j,prefix+"x",p==null?-999:p.getX());number(j,prefix+"y",p==null?-999:p.getY());number(j,prefix+"z",p==null?-999:p.getZ());
    }
    public static JsonObject read() {
        Minecraft mc=Minecraft.getInstance();JsonObject j=new JsonObject();
        ServerPlayerEntity sp=mc.player==null || mc.getIntegratedServer()==null ? null :
            mc.getIntegratedServer().getPlayerList().getPlayerByUUID(mc.player.getUniqueID());
        PlayerInteractionManager im=sp==null?null:sp.interactionManager;
        number(j,"place_available",sp==null?0:1);
        number(j,"place_count",im==null?0:im.minecraftRLPlaces);
        number(j,"place_last_tick",im==null?-1:im.minecraftRLLastPlaceTick);
        j.addProperty("place_last_block",im==null?"air":im.minecraftRLLastPlaced);
        position(j,"place_last_",im==null || im.minecraftRLPlaces==0?null:im.minecraftRLLastPlace);
        return j;
    }
}
