import numpy as np
import re

def vectorCosine(a,b) -> float:
    ''' the cosine of numpy vector a and b '''
    return np.dot(a,b) / (vectorLen(a) * vectorLen(b))

def vectorLen(v) -> float:
    ''' the length of numpy vector v '''
    return np.sqrt(np.dot(v,v))

def movieInfo(m):
    info = '''
           movie id | movie title | release date | video release date |
           IMDb URL | unknown | Action | Adventure | Animation |
           Children's | Comedy | Crime | Documentary | Drama | Fantasy |
           Film-Noir | Horror | Musical | Mystery | Romance | Sci-Fi |
           Thriller | War | Western |
           '''
    info = re.sub(r'\s+', '', info)
    info = info.split('|')
    desc = ' | '.join(m[0:3]) + ' | '
    for i in range(len(m)):
        if m[i] == '1':
            desc += info[i] + ','
    return desc

if __name__ == '__main__':
    v1 = np.array([1,1])
    v2 = np.array([1,0])
    print('v1 length: ',vectorLen(v1))
    print('cos: ', vectorCosine(v1,v2))
