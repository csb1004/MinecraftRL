"""Version-pinned vanilla recipes: material planning and exact grid validation."""
from collections import Counter
import json
import zipfile


def short(name):return name.removeprefix('minecraft:')


class RecipeCatalog:
    def __init__(self,recipes,tags):
        self.recipes=recipes;self.tags=tags
        self.by_output={short(r['result']['item']):key for key,r in recipes.items()}

    @classmethod
    def from_jar(cls,jar):
        ids=['stick','crafting_table','wooden_pickaxe','wooden_axe','oak_slab']
        ids += [kind+'_planks' for kind in ('oak','birch','spruce','jungle','acacia','dark_oak')]
        with zipfile.ZipFile(jar) as z:
            recipes={key:json.loads(z.read('data/minecraft/recipes/'+key+'.json')) for key in ids}
            tags={}
            def tag(name):
                if name in tags:return
                values=json.loads(z.read('data/minecraft/tags/items/'+short(name)+'.json'))['values']
                tags[name]=[]
                for value in values:
                    if value.startswith('#'):tag(value[1:]);tags[name]+=tags[value[1:]]
                    else:tags[name].append(value)
            for r in recipes.values():
                for ingredient in list(r.get('key',{}).values())+r.get('ingredients',[]):
                    if 'tag' in ingredient:tag(ingredient['tag'])
        return cls(recipes,tags)

    def choices(self,ingredient):
        if isinstance(ingredient,list):return sorted({item for part in ingredient for item in self.choices(part)})
        if 'item' in ingredient:return [short(ingredient['item'])]
        return [short(item) for item in self.tags[ingredient['tag']]]

    def dimensions(self,key):
        r=self.recipes[key]
        if r['type']=='minecraft:crafting_shaped':return max(map(len,r['pattern'])),len(r['pattern'])
        if r['type']=='minecraft:crafting_shapeless':return (2,2) if len(r['ingredients'])<=4 else (3,3)
        raise ValueError('unsupported recipe type')

    def ingredients(self,key):
        r=self.recipes[key]
        return [r['key'][char] for row in r['pattern'] for char in row if char!=' '] if 'pattern' in r else r['ingredients']

    def plan(self,inventory,goals):
        stock=Counter(inventory);steps=[]
        def ensure(item,count,trail=()):
            if stock[item]>=count:return
            if item in trail or item not in self.by_output:raise ValueError('missing material: '+item)
            key=self.by_output[item];r=self.recipes[key]
            while stock[item]<count:
                if len(steps)>128:raise ValueError('recipe plan too long')
                for ingredient in self.ingredients(key):
                    options=sorted(self.choices(ingredient),key=lambda x:(stock[x]<=0,x))
                    for option in options:
                        before=stock.copy();n=len(steps)
                        try:ensure(option,1,trail+(item,));stock[option]-=1;break
                        except ValueError:stock.clear();stock.update(before);del steps[n:]
                    else:raise ValueError('no available ingredient for '+key)
                steps.append(key);stock[item]+=r['result'].get('count',1)
        if any(max(self.dimensions(self.by_output[g]))>2 for g in goals):ensure('crafting_table',1)
        for goal in goals:ensure(goal,1)
        return steps,{k:v for k,v in stock.items() if v}

    def matches(self,key,width,items,counts):
        if width not in (2,3) or len(items)!=width*width or len(counts)!=len(items):return False
        if any(n not in (0,1) for n in counts):return False
        occupied=[items[i] for i,n in enumerate(counts) if n]
        r=self.recipes[key]
        if r['type']=='minecraft:crafting_shapeless':
            ingredients=r['ingredients']
            def match(rest,remaining):
                if not rest:return not remaining
                return any(item in self.choices(rest[0]) and match(rest[1:],remaining[:i]+remaining[i+1:])
                           for i,item in enumerate(remaining))
            return match(ingredients,occupied)
        pw,ph=self.dimensions(key)
        for flip in (False,True):
            for oy in range(width-ph+1):
                for ox in range(width-pw+1):
                    valid=True
                    for y in range(width):
                        for x in range(width):
                            char=' '
                            if ox<=x<ox+pw and oy<=y<oy+ph:
                                xx=pw-1-(x-ox) if flip else x-ox;char=r['pattern'][y-oy][xx]
                            i=y*width+x
                            if char==' ':valid &= counts[i]==0
                            else:valid &= counts[i]==1 and items[i] in self.choices(r['key'][char])
                    if valid:return True
        return False
