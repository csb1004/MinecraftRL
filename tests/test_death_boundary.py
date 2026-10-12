import sys
import unittest

SERVER='void loadOrCreateWorld() { String saveZipFile = getSaveFile(missionInit); loadWorld(saveZipFile); }'
RECORDER='public class PlayRecorder { void tick() { if (player == null || !player.isAlive()) { finishAndResetEpisode(); return; } } }'
SCREEN='void render() { // We have to respawn the player because otherwise code hangs\nthis.minecraft.player.respawnPlayer(); }'

class DeathBoundaryTests(unittest.TestCase):
    def patched(self):
        from minecraftrl.world_save_patch import patch_terminal_death
        return patch_terminal_death(SERVER,RECORDER,SCREEN)

    def test_terminal_setting_precedes_saved_or_new_world_branch(self):
        server,_,_=self.patched()
        self.assertLess(server.index('setTerminalOnDeath'),server.index('getSaveFile'))
        self.assertIn('this.doneOnDeath = isDoneOnDeath(missionInit)',server)
        self.assertIn('setHasPlayerRespawned(false)',server)

    def test_dead_frame_is_published_after_inputs_clear_before_old_reset(self):
        _,recorder,_=self.patched()
        head=recorder[:recorder.index('if (player == null || !player.isAlive())')]
        self.assertIn('terminalOnDeath && player != null && !player.isAlive()',head)
        self.assertLess(head.index('unPressAllKeys()'),head.index('recordTickImpl()'))
        self.assertIn('clearMouseKeyboardState()',head)
        self.assertIn('volatile boolean terminalOnDeath',recorder)

    def test_autorespawn_is_guarded_without_removing_normal_mode(self):
        _,_,screen=self.patched()
        self.assertLess(screen.index('isTerminalOnDeath()'),screen.index('respawnPlayer()'))
        self.assertIn('return;',screen)
        self.assertEqual(screen.count('respawnPlayer()'),1)

    def test_unknown_upstream_is_rejected_for_every_patch_input(self):
        from minecraftrl.world_save_patch import patch_terminal_death
        for index in range(3):
            sources=[SERVER,RECORDER,SCREEN];sources[index]='unexpected'
            with self.subTest(index=index),self.assertRaises(ValueError):patch_terminal_death(*sources)

    @unittest.skipUnless(sys.platform=='linux','MineRL specification is installed in WSL')
    def test_death_handler_is_explicit_and_opt_in(self):
        from minecraftrl.workshop_monitor import workshop_survival
        normal=workshop_survival()
        terminal=workshop_survival(terminate_on_death=True)
        self.assertNotIn('done_on_death',[x.to_string() for x in normal.create_agent_start()])
        self.assertEqual([x.to_string() for x in terminal.create_agent_start()].count('done_on_death'),1)
