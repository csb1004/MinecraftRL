import json
from pathlib import Path
import tempfile
import unittest

import test_recovery_draft as fixtures


class RecoveryExchangeTests(unittest.TestCase):
    def make(self):
        from minecraftrl.recovery_exchange import FileRecoveryProvider
        temp=tempfile.TemporaryDirectory();self.addCleanup(temp.cleanup)
        path=Path(temp.name)/'one-request'
        return FileRecoveryProvider(path),path

    def test_request_reply_identity_and_one_use(self):
        provider,path=self.make();packet=fixtures.RecoveryDraftTests().packet()
        provider.start(packet)
        self.assertEqual(json.loads((path/'request.json').read_text())['packet'],packet)
        self.assertIsNone(provider.poll())
        from minecraftrl.recovery_draft import validate_draft
        reply=validate_draft(fixtures.RecoveryDraftTests().reply(),packet)
        (path/'response.json').write_text(json.dumps({'request_id':packet['request_id'],'proposal':reply}))
        self.assertEqual(provider.poll(),reply)
        with self.assertRaises(RuntimeError):provider.start(packet)

    def test_closed_provider_ignores_late_reply(self):
        provider,path=self.make();packet=fixtures.RecoveryDraftTests().packet();provider.start(packet)
        provider.close()
        (path/'response.json').write_text('{"malicious":"late"}')
        self.assertIsNone(provider.poll())
        closed=json.loads((path/'closed.json').read_text())
        self.assertEqual(closed['request_id'],packet['request_id'])
        self.assertTrue(provider.closed)

    def test_existing_exchange_is_not_overwritten(self):
        provider,path=self.make();path.mkdir();(path/'request.json').write_text('existing')
        with self.assertRaises(FileExistsError):provider.start(fixtures.RecoveryDraftTests().packet())
        self.assertEqual((path/'request.json').read_text(),'existing')

    def test_wrong_or_extra_reply_fields_rejected(self):
        for reply in ({'request_id':'wrong','proposal':{}},{'request_id':'probe-1','proposal':{},'reward':1}):
            provider,path=self.make();provider.start(fixtures.RecoveryDraftTests().packet())
            (path/'response.json').write_text(json.dumps(reply))
            with self.assertRaises(ValueError):provider.poll()

    def test_close_before_start_does_not_create_exchange(self):
        provider,path=self.make();provider.close()
        self.assertFalse(path.exists())
        with self.assertRaises(RuntimeError):provider.start(fixtures.RecoveryDraftTests().packet())
