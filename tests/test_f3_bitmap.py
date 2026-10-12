import unittest
from pathlib import Path
import numpy as np
from minecraftrl.f3_bitmap import read_f3_bitmap,load_font


class F3BitmapTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.root=Path(__file__).resolve().parents[1]
        jar=cls.root/'.runtime/steve1/lib/python3.10/site-packages/minerl/MCP-Reborn/build/libs/mcprec-6.13.jar'
        if not jar.exists():raise unittest.SkipTest('pinned Minecraft font not installed')
        cls.font=load_font(jar)

    def test_real_capture_matches_visible_text_without_telemetry_input(self):
        from PIL import Image
        path=self.root/'results/spatial-exploration-20261010/f3-probe/f3_visible.png'
        if not path.exists():self.skipTest('actual F3 capture unavailable')
        value=read_f3_bitmap(np.asarray(Image.open(path).convert('RGB')),self.font)
        self.assertEqual(value['position'],[229.5,201.,-785.3])
        with self.assertRaises(ValueError):read_f3_bitmap(np.zeros((360,640,3),dtype=np.uint8),self.font)
        doubled=np.concatenate([np.asarray(Image.open(path).convert('RGB'))]*2,axis=0)
        with self.assertRaises(ValueError):read_f3_bitmap(doubled,self.font)
        closed=path.with_name('f3_closed.png')
        with self.assertRaises(ValueError):read_f3_bitmap(np.asarray(Image.open(closed).convert('RGB')),self.font)

    def test_font_roundtrip_scales_signs_and_damaged_digit(self):
        for scale in (1,2,3,4):
            text='XYZ: -12.500 / 65.00000 / 2.250'
            glyphs=np.concatenate([self.font['glyphs'][c] for c in text],axis=1)
            glyphs=np.repeat(np.repeat(glyphs,scale,axis=0),scale,axis=1)
            pixels=np.zeros((80*scale,400*scale,3),dtype=np.uint8)
            y,x=10*scale,2*scale
            pixels[y:y+glyphs.shape[0],x:x+glyphs.shape[1]]=glyphs[:,:,None]*221
            with self.subTest(scale=scale):
                self.assertEqual(read_f3_bitmap(pixels,self.font)['position'],[-12.5,65.,2.25])
                # Damage the first digit while preserving the prefix; never guess a repair.
                digit=x+sum(self.font['glyphs'][c].shape[1] for c in 'XYZ: -')*scale
                pixels[y,digit]=221
                with self.assertRaises(ValueError):read_f3_bitmap(pixels,self.font)


if __name__=='__main__':unittest.main()
