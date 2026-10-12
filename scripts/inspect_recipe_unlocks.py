"""Read exact-version recipe unlock rules and actual saved player recipe books."""
import argparse
import json
import zipfile
from pathlib import Path
from minecraftrl.saved_inventory import read_nbt
from minecraftrl.world_store import digest


def inspect(archive,jar):
    targets=('oak_planks','stick','crafting_table','wooden_pickaxe','oak_slab')
    result={'archive':str(archive),'archive_sha256':digest(archive),'jar_sha256':digest(jar),
            'players':[],'advancements':{},'rules':{},'recipes':{}}
    with zipfile.ZipFile(archive) as z:
        for name in z.namelist():
            if name.endswith('/level.dat') or ('/playerdata/' in name and name.endswith('.dat')):
                value=read_nbt(z.read(name))
                if name.endswith('/level.dat'):
                    data=value['Data'];result['version']=data.get('Version');result['gamerules']=data.get('GameRules')
                    value=data['Player']
                book=value.get('recipeBook',{})
                result['players'].append({'file':name,'recipe_book':book,
                    'target_unlocks':{item:'minecraft:'+item in book.get('recipes',[]) for item in targets}})
            if '/advancements/' in name and name.endswith('.json'):
                result['advancements'][name]={k:v for k,v in json.loads(z.read(name)).items()
                                             if k.split('/')[-1] in targets}
    with zipfile.ZipFile(jar) as z:
        result['jar_version']=json.loads(z.read('version.json'))
        for name in z.namelist():
            if name.startswith('data/minecraft/advancements/recipes/') and name.rsplit('/',1)[-1][:-5] in targets:
                result['rules'][name]=json.loads(z.read(name))
        for item in targets:result['recipes'][item]=json.loads(z.read('data/minecraft/recipes/'+item+'.json'))
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--archive',type=Path,required=True)
    p.add_argument('--jar',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();result=inspect(a.archive,a.jar)
    a.output.write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({'output':str(a.output),'version':result['version'],
                      'players':[x['target_unlocks'] for x in result['players']]}))
