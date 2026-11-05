package net.minecraft.client.gui.screen.inventory;

import com.google.gson.Gson;
import com.google.gson.JsonObject;
import com.minerl.multiagent.RandomHelper;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.nio.file.StandardOpenOption;
import java.security.MessageDigest;
import net.minecraft.client.Minecraft;
import net.minecraft.entity.player.ServerPlayerEntity;
import net.minecraft.world.GameRules;

/** Opt-in restoration at a pinned startup boundary; no policy input or simulation step. */
public final class StartupRandomTicks {
    private static JsonObject request;
    private static Path directory;
    private static String requestHash;
    private static Object world;
    private static boolean restored,advanced;
    private static int resumeTick,restoreSpeed;

    private static String hash(byte[] bytes) throws Exception {
        StringBuilder out=new StringBuilder();
        for(byte b:MessageDigest.getInstance("SHA-256").digest(bytes))out.append(String.format("%02x",b & 255));
        return out.toString();
    }
    private static int integer(String key,int maximum) {
        String value=request.get(key).getAsString();
        if(!value.matches("0|[1-9][0-9]*"))throw new IllegalArgumentException("Invalid startup integer: "+key);
        int result=Integer.parseInt(value);
        if(result>maximum)throw new IllegalArgumentException("Startup integer out of range: "+key);
        return result;
    }
    private static void load() throws Exception {
        String value=System.getenv("MINECRAFTRL_STARTUP_RANDOM_TICKS");
        if(value==null || value.isEmpty())throw new IllegalStateException("Explicit startup control request required");
        Path path=Paths.get(value).toAbsolutePath().normalize();
        if(!Files.isRegularFile(path) || Files.size(path)>16384)throw new IllegalStateException("Invalid startup request file");
        byte[] bytes=Files.readAllBytes(path);requestHash=hash(bytes);
        request=new Gson().fromJson(new String(bytes,StandardCharsets.UTF_8),JsonObject.class);
        if(!request.get("schema").getAsString().equals("startup_random_tick_request_v1"))
            throw new IllegalStateException("Unsupported startup control schema");
        resumeTick=integer("resume_client_tick",80);restoreSpeed=integer("restore_speed",4096);
        if(resumeTick!=80 || integer("paused_speed",0)!=0)throw new IllegalStateException("Unsupported startup boundary");
        directory=path.getParent();
    }
    private static void record(String name,Minecraft mc,ServerPlayerEntity sp,long randomBefore,
            int tickBefore,int clientBefore,long worldBefore) throws Exception {
        long randomAfter=RandomHelper.getSeed(RandomHelper.getRandom());
        if(randomBefore!=randomAfter || tickBefore!=mc.getIntegratedServer().getTickCounter() ||
                clientBefore!=mc.player.ticksExisted || worldBefore!=sp.getServerWorld().getGameTime())
            throw new IllegalStateException("Startup restoration advanced RNG or simulation");
        JsonObject j=new JsonObject();
        j.addProperty("schema","startup_random_tick_receipt_v1");j.addProperty("request_sha256",requestHash);
        j.addProperty("restored",restored);j.addProperty("restore_speed",restoreSpeed);
        j.addProperty("current_speed",sp.getServerWorld().getGameRules().getInt(GameRules.RANDOM_TICK_SPEED));
        j.addProperty("client_tick",mc.player.ticksExisted);j.addProperty("server_tick",tickBefore);
        j.addProperty("world_time",sp.getServerWorld().getGameTime());
        j.addProperty("server_tick_before",tickBefore);j.addProperty("client_tick_before",clientBefore);
        j.addProperty("world_time_before",worldBefore);
        j.addProperty("random_state_before",randomBefore);j.addProperty("random_state_after",randomAfter);
        j.addProperty("extra_simulation_ticks",0);j.addProperty("input_commands",0);
        Files.write(directory.resolve(name),j.toString().getBytes(StandardCharsets.UTF_8),StandardOpenOption.CREATE_NEW);
    }
    public static void capture() {
        try {
            Minecraft mc=Minecraft.getInstance();
            if(mc.player==null || mc.getIntegratedServer()==null)return;
            ServerPlayerEntity sp=mc.getIntegratedServer().getPlayerList().getPlayerByUUID(mc.player.getUniqueID());
            if(sp==null)return;
            if(request==null)load();
            if(world==null)world=sp.getServerWorld();
            if(world!=sp.getServerWorld())throw new IllegalStateException("Startup request cannot be reused for another world");
            int tick=mc.player.ticksExisted;
            GameRules rules=sp.getServerWorld().getGameRules();
            if(!restored) {
                if(rules.getInt(GameRules.RANDOM_TICK_SPEED)!=0)throw new IllegalStateException("Startup world was not paused");
                if(tick>resumeTick)throw new IllegalStateException("Missed startup restoration boundary");
                if(tick==resumeTick) {
                    long randomBefore=RandomHelper.getSeed(RandomHelper.getRandom());
                    int serverBefore=mc.getIntegratedServer().getTickCounter();
                    long worldBefore=sp.getServerWorld().getGameTime();
                    GameRules.IntegerValue replacement=new GameRules().get(GameRules.RANDOM_TICK_SPEED);
                    if(!replacement.parseIntValue(Integer.toString(restoreSpeed)))throw new IllegalStateException("Invalid restore speed");
                    rules.get(GameRules.RANDOM_TICK_SPEED).changeValue(replacement,mc.getIntegratedServer());
                    restored=true;
                    if(rules.getInt(GameRules.RANDOM_TICK_SPEED)!=restoreSpeed)throw new IllegalStateException("Rule restoration failed");
                    record("startup-random-ticks-restored.json",mc,sp,randomBefore,serverBefore,tick,worldBefore);
                }
            } else {
                if(rules.getInt(GameRules.RANDOM_TICK_SPEED)!=restoreSpeed)throw new IllegalStateException("Restored rule changed unexpectedly");
                if(!advanced && tick>resumeTick) {
                    if(tick!=resumeTick+1)throw new IllegalStateException("Missing first post-startup capture");
                    record("startup-random-ticks-first-step.json",mc,sp,
                        RandomHelper.getSeed(RandomHelper.getRandom()),mc.getIntegratedServer().getTickCounter(),
                        tick,sp.getServerWorld().getGameTime());
                    advanced=true;
                }
            }
        } catch(Exception e) {throw new IllegalStateException("MinecraftRL startup random tick control failed",e);}
    }
}
