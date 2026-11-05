package net.minecraft.client.gui.screen.inventory;

import com.google.gson.Gson;
import com.google.gson.JsonArray;
import com.google.gson.JsonElement;
import com.google.gson.JsonObject;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.security.MessageDigest;
import net.minecraft.block.Block;
import net.minecraft.client.Minecraft;
import net.minecraft.entity.player.ServerPlayerEntity;
import net.minecraft.item.Item;
import net.minecraft.item.ItemStack;
import net.minecraft.item.Items;
import net.minecraft.server.MinecraftServer;
import net.minecraft.util.Direction;
import net.minecraft.util.ResourceLocation;
import net.minecraft.util.math.BlockPos;
import net.minecraft.util.registry.Registry;
import net.minecraft.world.server.ServerWorld;

/** Opt-in demo world preparation: places configured blocks once, on the server thread, relative to the start pose. */
public final class DemoFixture {
    // 0 none requested, 1 reserved, 2 applied, -1 error. Read by DemoObservation; never reset in a session.
    private static volatile int state=0,blocks=0,originY=Integer.MIN_VALUE;
    private static volatile long sha52=0;
    public static int state() {return state;}
    public static int blocks() {return blocks;}
    /** Start block height the fixture was placed around (pit depth reference). */
    public static int originY() {return originY;}
    public static long sha52() {return sha52;}

    public static void tick() {
        if(state!=0)return;
        String value=System.getenv("MINECRAFTRL_DEMO_FIXTURE");
        if(value==null || value.isEmpty())return;
        Minecraft mc=Minecraft.getInstance();MinecraftServer server=mc.getIntegratedServer();
        if(mc.player==null || server==null)return;
        ServerPlayerEntity sp=server.getPlayerList().getPlayerByUUID(mc.player.getUniqueID());
        if(sp==null)return;
        state=1;
        server.execute(() -> apply(value,sp));
    }
    private static int offset(JsonElement value) {
        int result=value.getAsInt();
        if(Math.abs(result)>16)throw new IllegalArgumentException("Fixture offset out of range");
        return result;
    }
    private static BlockPos at(BlockPos origin,Direction forward,int dx,int dy,int dz) {
        // dx: right of the start facing, dz: forward, dy: up from the start block.
        return origin.offset(forward,dz).offset(forward.rotateY(),dx).up(dy);
    }
    private static Block block(String name) {
        ResourceLocation key=new ResourceLocation("minecraft",name);
        if(!Registry.BLOCK.containsKey(key))throw new IllegalArgumentException("Unknown fixture block: "+name);
        return Registry.BLOCK.getOrDefault(key);
    }
    private static void apply(String value,ServerPlayerEntity sp) {
        try {
            Path path=Paths.get(value).toAbsolutePath().normalize();
            if(!Files.isRegularFile(path) || Files.size(path)>65536)throw new IllegalStateException("Invalid fixture file");
            byte[] bytes=Files.readAllBytes(path);
            JsonObject cfg=new Gson().fromJson(new String(bytes,StandardCharsets.UTF_8),JsonObject.class);
            if(!cfg.get("schema").getAsString().equals("iron_demo_fixture_v1"))throw new IllegalStateException("Unsupported fixture schema");
            Direction forward=cfg.get("rotate_with_facing").getAsBoolean()?sp.getHorizontalFacing():Direction.SOUTH;
            BlockPos origin=sp.getPosition();ServerWorld world=sp.getServerWorld();int placed=0;
            if(cfg.has("fill"))for(JsonElement e:cfg.getAsJsonArray("fill")) {
                JsonObject r=e.getAsJsonObject();JsonArray from=r.getAsJsonArray("from"),to=r.getAsJsonArray("to");
                Block b=block(r.get("block").getAsString());int[] low=new int[3],high=new int[3];
                for(int i=0;i<3;i++) {
                    low[i]=Math.min(offset(from.get(i)),offset(to.get(i)));high[i]=Math.max(offset(from.get(i)),offset(to.get(i)));
                }
                for(int dx=low[0];dx<=high[0];dx++)for(int dy=low[1];dy<=high[1];dy++)for(int dz=low[2];dz<=high[2];dz++) {
                    world.setBlockState(at(origin,forward,dx,dy,dz),b.getDefaultState(),3);placed++;
                }
            }
            for(JsonElement e:cfg.getAsJsonArray("blocks")) {
                JsonObject b=e.getAsJsonObject();
                world.setBlockState(at(origin,forward,offset(b.get("dx")),offset(b.get("dy")),offset(b.get("dz"))),
                    block(b.get("block").getAsString()).getDefaultState(),3);
                placed++;
            }
            for(JsonElement e:cfg.getAsJsonArray("give")) {
                JsonObject g=e.getAsJsonObject();ResourceLocation key=new ResourceLocation("minecraft",g.get("item").getAsString());
                Item item=Registry.ITEM.getOrDefault(key);int count=g.get("count").getAsInt();
                if(!Registry.ITEM.containsKey(key) || item==Items.AIR || count<1 || count>64)throw new IllegalArgumentException("Invalid fixture item");
                if(!sp.inventory.addItemStackToInventory(new ItemStack(item,count)))throw new IllegalStateException("Fixture item did not fit");
            }
            long digest=0;byte[] hash=MessageDigest.getInstance("SHA-256").digest(bytes);
            // First 13 hex digits (52 bits): exact in the float64 observation channel.
            for(int i=0;i<7;i++)digest=(digest<<8)|(hash[i] & 255);
            sha52=digest>>>4;blocks=placed;originY=origin.getY();state=2;
            System.out.println("[MinecraftRL] demo fixture applied: "+placed+" blocks facing "+forward);
        } catch(Exception error) {
            state=-1;System.out.println("[MinecraftRL] demo fixture failed: "+error);
        }
    }
}
