import smiles
import latent_dict

smiles_list = smiles.get_smile_list()

print len(smiles_list)

print len(set(smiles_list))

smile_latent_dict = latent_dict.get_latent_dict(smiles_list)

print type (smile_latent_dict), len(smile_latent_dict)