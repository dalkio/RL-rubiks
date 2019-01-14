dim = 3
colors = ('blue', 'green', 'orange', 'red', 'white', 'yellow')
sides = ('r', 'l', 'u', 'd', 'f', 'b')
directions = ('d', 'i')
connexions = {
    'r': {
        'sides': ('f', 'd', 'b', 'u', 'f'), 
        'edges': ('r', 'r', 'l', 'r', 'r'), 
        'inversions': {('r', 'l'), ('l', 'r')}
    },
    'l': {
        'sides': ('f', 'u', 'b', 'd', 'f'), 
        'edges': ('l', 'l', 'r', 'l', 'l'),
        'inversions': {('l', 'r'), ('r', 'l')}
    },
    'u': {
        'sides': ('f', 'r', 'b', 'l', 'f'), 
        'edges': ('u', 'u', 'u', 'u', 'u'),
        'inversions': {}
    },
    'd': {
        'sides': ('f', 'l', 'b', 'r', 'f'), 
        'edges': ('d', 'd', 'd', 'd', 'd'),
        'inversions': {}
    },
    'f': {
        'sides': ('u', 'l', 'd', 'r', 'u'), 
        'edges': ('d', 'r', 'u', 'l', 'd'),
        'inversions': {('d', 'r'), ('r', 'd'), ('u', 'l'), ('l', 'u')}
    },
    'b': {
        'sides': ('d', 'l', 'u', 'r', 'd'), 
        'edges': ('d', 'l', 'u', 'r', 'd'),
        'inversions': {('r', 'd'), ('d', 'r'), ('l', 'u'), ('u', 'l')}
    }
}