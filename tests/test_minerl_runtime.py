import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from minecraftrl.minerl_runtime import prepare_jar_cache


class RuntimeJarTests(unittest.TestCase):
    def test_persistent_player_identity_survives_process_setup(self):
        from minecraftrl.minerl_runtime import configure_player_identity
        with tempfile.TemporaryDirectory() as tmp, patch.dict(os.environ):
            game=Path(tmp); launcher=game/'launchClient.sh'
            launcher.write_text('java -Xmx$maxMem -jar "$fatjar" --envPort=$port\n')
            first=configure_player_identity(game,'companion')
            os.environ.pop('MINECRAFTRL_PLAYER_UUID')
            second=configure_player_identity(game,'companion')
            self.assertEqual(first,second)
            self.assertEqual(os.environ['MINECRAFTRL_PLAYER_UUID'],first)
            self.assertNotEqual(first,configure_player_identity(game,'independent'))
            self.assertEqual(launcher.read_text().count('--username MinecraftRL'),1)

    def test_cache_preserves_bytes_and_game_working_directory(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); game=root/'game'; output=root/'results'; output.mkdir()
            jar=game/'build/libs/mcprec-6.13.jar'; jar.parent.mkdir(parents=True)
            jar.write_bytes(b'immutable fixture jar bytes')
            launcher=game/'launchClient.sh'
            launcher.write_text('fatjar=build/libs/mcprec-6.13.jar\njava -jar $fatjar --envPort=9780\n')
            for index in range(2):
                scratch=root/str(index); scratch.mkdir()
                with patch.dict(os.environ), patch('minecraftrl.minerl_runtime.tempfile.mkdtemp',return_value=str(scratch)):
                    copied=prepare_jar_cache(game,output)
                    self.assertEqual(copied.read_bytes(),jar.read_bytes())
                    self.assertEqual(os.environ['MINECRAFTRL_RUNTIME_JAR'],str(copied))
            self.assertEqual(launcher.read_text().count('MINECRAFTRL_RUNTIME_JAR'),1)
