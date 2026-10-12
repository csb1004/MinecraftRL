"""Pinned MineRL save-on-quit fix, compiled into a separate session-only JAR."""
import hashlib
import json
from pathlib import Path
import subprocess
import uuid
import zipfile
from .memory import atomic_json, file_lock
from .world_store import digest

UPSTREAM_SHA256='f7a1e0aa8bee8a0ff9479856e807be6d8aa30373db8934d5b093342690ae9a48'
CLASS='net/minecraft/client/ReplaySender'


def patch_source(source):
    marker='mc.world.sendQuittingDisconnectingPacket();'
    if source.count(marker)!=1 or 'public void stop()' not in source:
        raise ValueError('unexpected_upstream_save_source')
    return source.replace(marker,'''                          // MinecraftRL: synchronous server has no shutdown save barrier.
                          // Flush actual player + chunks before disconnect, on the render/server thread.
                          if (mc.getIntegratedServer() != null) {
                              mc.getIntegratedServer().getPlayerList().saveAllPlayerData();
                              mc.getIntegratedServer().save(false, true, true);
                              System.out.println("[MinecraftRL] player and world save barrier completed");
                          }
'''+marker)


def patch_world_identity(source):
    marker='String worldName = "mcpworld" + RandomHelper.getRandomHexString();'
    if source.count(marker)!=1: raise ValueError('unexpected_world_identity_source')
    # Consume the same game RNG draw; only the storage identifier is independent.
    return source.replace(marker,marker+'''
        worldName += "-" + java.util.UUID.randomUUID().toString();
        if (java.nio.file.Files.exists(mc.getSaveLoader().getSavesDir().resolve(worldName))) {
            throw new IllegalStateException("MinecraftRL fresh world path already exists");
        }
''')


def patch_terminal_death(server,recorder,screen):
    load='String saveZipFile = getSaveFile(missionInit);'
    declaration='public class PlayRecorder {'
    dead='if (player == null || !player.isAlive()) {'
    respawn='// We have to respawn the player because otherwise code hangs'
    for source,marker in ((server,load),(recorder,declaration),(recorder,dead),(screen,respawn)):
        if source.count(marker)!=1:raise ValueError('unexpected_terminal_death_source')
    server=server.replace(load,'''this.doneOnDeath = isDoneOnDeath(missionInit);
        PlayRecorder.getInstance().setTerminalOnDeath(this.doneOnDeath);
        Minecraft.getInstance().setHasPlayerRespawned(false);
        '''+load)
    recorder=recorder.replace(declaration,declaration+'''
    private volatile boolean terminalOnDeath = false;
    public void setTerminalOnDeath(boolean enabled) { terminalOnDeath = enabled; }
    public boolean isTerminalOnDeath() { return terminalOnDeath; }
''')
    recorder=recorder.replace(dead,'''if (terminalOnDeath && player != null && !player.isAlive()) {
            if (!recording) start();
            mc.mouseHelper.setHumanInput(false);
            clearMouseKeyboardState();
            net.minecraft.client.settings.KeyBinding.unPressAllKeys();
            // Publish the terminal frame; ReplaySender then waits for shutdown, not another action.
            recordTickImpl();
            return;
        }
        '''+dead)
    screen=screen.replace(respawn,'''// A terminal RL mission must preserve the dead player until observation/shutdown.
      if (com.minerl.multiagent.recorder.PlayRecorder.getInstance().isTerminalOnDeath()) return;
      '''+respawn)
    return server,recorder,screen


def patch_mining_observation(source):
    marker='public class PlayerInteractionManager {'
    if source.count(marker)!=1:raise ValueError('unexpected_mining_observation_source')
    return source.replace(marker,marker+'''
   // Read-only server evidence, sampled at the existing recorder boundary.
   public BlockPos minecraftRLMiningTarget() {
      if (this.receivedFinishDiggingPacket) return this.delayedDestroyPos;
      return this.isDestroyingBlock ? this.destroyPos : null;
   }
''')


def build_save_jar(root,minecraft_dir,*,craft_monitor=False,workshop_monitor=False):
    if workshop_monitor and not craft_monitor:raise ValueError('workshop monitor requires craft monitor')
    root=Path(root); minecraft_dir=Path(minecraft_dir)
    upstream=minecraft_dir/'build/libs/mcprec-6.13.jar'
    if digest(upstream)!=UPSTREAM_SHA256: raise ValueError('unsupported_upstream_world_jar')
    source=minecraft_dir/('src/main/java/'+CLASS+'.java')
    patched=patch_source(source.read_text(encoding='utf-8'))
    server=minecraft_dir/'src/main/java/com/minerl/multiagent/env/EnvServer.java'
    server_text=patch_world_identity(server.read_text(encoding='utf-8'))
    extra={'EnvServer.java':server_text}
    if craft_monitor:
        marker='infoJson.addProperty("isGuiOpen", mc.currentScreen != null);'
        if server_text.count(marker)!=1: raise ValueError('unexpected_craft_monitor_source')
        extra['EnvServer.java']=server_text.replace(marker,marker+'\n        infoJson.add("craft_gui", net.minecraft.client.gui.screen.inventory.CraftGuiObservation.read());')
        extra['CraftGuiObservation.java']=Path(__file__).with_name('CraftGuiObservation.java').read_text(encoding='utf-8')
        if workshop_monitor:
            extra['EnvServer.java']=extra['EnvServer.java'].replace(marker,marker+'\n        infoJson.add("workshop", net.minecraft.client.gui.screen.inventory.WorkshopObservation.snapshot());')
            extra['WorkshopObservation.java']=Path(__file__).with_name('WorkshopObservation.java').read_text(encoding='utf-8')
            interaction=minecraft_dir/'src/main/java/net/minecraft/server/management/PlayerInteractionManager.java'
            extra['PlayerInteractionManager.java']=patch_mining_observation(interaction.read_text(encoding='utf-8'))
            recorder=minecraft_dir/'src/main/java/com/minerl/multiagent/recorder/PlayRecorder.java'
            recorder_text=recorder.read_text(encoding='utf-8');capture='tickCounter++;'
            death_screen=minecraft_dir/'src/main/java/net/minecraft/client/gui/screen/DeathScreen.java'
            extra['EnvServer.java'],recorder_text,extra['DeathScreen.java']=patch_terminal_death(
                extra['EnvServer.java'],recorder_text,death_screen.read_text(encoding='utf-8'))
            if recorder_text.count(capture)!=1:raise ValueError('unexpected_workshop_snapshot_boundary')
            # Publish on the render/server thread BEFORE announcing the next frame.
            # The socket thread neither loads chunks nor races snapshot publication.
            extra['PlayRecorder.java']=recorder_text.replace(capture,
                'net.minecraft.client.gui.screen.inventory.WorkshopObservation.capture();\n        '+capture)
    patch_hash=hashlib.sha256((patched+''.join(extra.values())).encode()).hexdigest()
    base=root/'.runtime/world-save'; base.mkdir(parents=True,exist_ok=True)
    with file_lock(base/'build.lock'):
        receipt=base/(patch_hash+'.json')
        if receipt.exists():
            value=json.loads(receipt.read_text()); artifact=base/value['artifact']
            if digest(artifact)!=value['sha256']: raise ValueError('modified_world_save_jar')
            return artifact
        build=base/uuid.uuid4().hex; build.mkdir(); classes=build/'classes'; classes.mkdir()
        java=build/'ReplaySender.java'; java.write_text(patched,encoding='utf-8')
        sources=[str(java)]
        for name,content in extra.items():
            path=build/name; path.write_text(content,encoding='utf-8'); sources.append(str(path))
        result=subprocess.run(['javac','-encoding','UTF-8','-source','8','-target','8','-cp',str(upstream),
                               '-d',str(classes),*sources],capture_output=True,text=True,timeout=120)
        (build/'javac.log').write_text(result.stdout+result.stderr,encoding='utf-8')
        if result.returncode: raise RuntimeError('World save JAR compilation failed: '+str(build/'javac.log'))
        replacements={p.relative_to(classes).as_posix():p for p in classes.rglob('*.class')}
        if CLASS+'.class' not in replacements: raise RuntimeError('Missing compiled save barrier')
        artifact=base/(patch_hash+'.jar'); temporary=build/'patched.jar'
        with zipfile.ZipFile(upstream) as src, zipfile.ZipFile(temporary,'x') as dst:
            for entry in src.infolist():
                dst.writestr(entry,replacements[entry.filename].read_bytes() if entry.filename in replacements else src.read(entry))
            for name,path in replacements.items():
                if name not in src.namelist(): dst.writestr(name,path.read_bytes())
        with zipfile.ZipFile(temporary) as check:
            if check.testzip(): raise RuntimeError('Invalid patched world save JAR')
        temporary.replace(artifact)
        atomic_json(receipt,{'artifact':artifact.name,'sha256':digest(artifact),'upstream_sha256':UPSTREAM_SHA256,
                            'source_sha256':digest(source),'patched_source_sha256':patch_hash,
                            'replaced_classes':sorted(replacements),'scope':'save barrier + fresh world storage identity' +
                            (' + read-only craft GUI monitor' if craft_monitor else '') +
                            (' + read-only workshop telemetry + opt-in terminal death handshake' if workshop_monitor else '')})
        return artifact
