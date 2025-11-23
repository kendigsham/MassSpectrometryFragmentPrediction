



def readfile():
    smilesnames = []
    with open('smilesnames.txt','r') as rr:
        for name in rr:
            smilesnames.append(name)
    return smilesnames

def getchar():
    charalist=[]
    with open('character.txt','r') as rrr:
        for line in rrr:
            temp=list(line)
            charalist.append(temp[1])
    return charalist

print('hihi')

smilelist=readfile()

charlist=getchar()

for i in charlist:
    print(i)

# for name in smilelist:
#     print(name)

# canonicalised_smiles = [Chem.MolToSmiles(Chem.MolFromSmiles(str(x))) for x in smilelist]
# input_array = get_input_arr(canonicalised_smiles, charset)
# latent_reps = encode(model, input_array)