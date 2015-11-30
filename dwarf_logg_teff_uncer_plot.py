import numpy as np
from astropy.io import fits as pyfits
import pandas as pd
import matplotlib
from matplotlib import pyplot as plt

caldata = pyfits.open('cal.fits')
data = caldata[1].data

M67_mask = data['FIELD'] == 'M67'
pleiades_mask = data['FIELD'] == 'Pleiades'

m67_ple = M67_mask | pleiades_mask

SNR_100 = data['SNR'] > 100
SNR_80 = data['SNR'] > 80

bins_logg = np.arange(1.5, 5.1, 0.5)
bins_teff = np.arange(3000, 6501, 500)

element_labels = ['[C/H]',  '[CI/H]', '[N/H]',  '[O/H]',  '[Na/H]',
                  '[Mg/H]', '[Al/H]', '[Si/H]', '[P/H]',  '[S/H]',
                  '[K/H]',  '[Ca/H]', '[Ti/H]', '[TiII/H]', '[V/H]',
                  '[Cr/H]', '[Mn/H]', '[Fe/H]', '[Co/H]', '[Ni/H]',
                  '[Cu/H]', '[Ge/H]', '[Ce/H]', '[Rb/H]', '[Y/H]',
                  '[Nd/H]']


param_cols = ['teff', 'logg']


data_dict = {}
data_dict['field'] = data['field']
data_dict['teff_var'] = np.array(data['FPARAM_COV'][:,0,0], dtype=float)
data_dict['logg_var'] = np.array(data['FPARAM_COV'][:,1,1], dtype=float)
data_dict['logg'] = np.array(data['FPARAM'][:,1], dtype=float)
data_dict['teff'] = np.array(data['FPARAM'][:,0], dtype=float)
for i, label in enumerate(element_labels):
    data_dict[label] = np.array(data['FELEM'][:,i], dtype=float)

df_data = pd.DataFrame(data_dict)
df_m67 = df_data[df_data['field'].str.match('M67')]
df_ple = df_data[df_data['field'].str.match('Pleiades')]

mean_elem_m67 = np.array([df_m67[label].mean() for label in element_labels])
mean_elem_ple = np.array([df_ple[label].mean() for label in element_labels])

#calculate residuals and make new columns
for i, label in enumerate(element_labels):
    resid_m67 = df_m67[label]- mean_elem_m67[i]
    resid_ple = df_ple[label]- mean_elem_ple[i]
    df_m67['res_'+label] = pd.Series(resid_m67, index=df_m67.index)
    df_ple['res_'+label] = pd.Series(resid_ple, index=df_ple.index)


grp_m67 = df_m67.groupby([pd.cut(df_m67['teff'], bins_teff),
                          pd.cut(df_m67['logg'], bins_logg)])

grp_ple = df_ple.groupby([pd.cut(df_ple['teff'], bins_teff),
                          pd.cut(df_ple['logg'], bins_logg)])


df_both = pd.concat((df_m67, df_ple))
grp_both = df_both.groupby([pd.cut(df_both['teff'], bins_teff),
                          pd.cut(df_both['logg'], bins_logg)])


def loggplot():
    fig = plt.figure()
    ax = fig.add_subplot(111)
    im = ax.scatter(df_both['teff'], df_both['logg'], c=np.sqrt(df_both['logg_var']), vmin=0, vmax=0.1)
    ax.set_xlim(6500,2500)
    ax.set_ylim(5.2, 1.5)
    #ax.text(6000, 2.0, element_labels[val])
    cb = plt.colorbar(im)
    cb.set_label('log(g) uncertainty')

loggplot()
plt.savefig('logg_uncer.png', format='png')
plt.savefig('logg_uncer.eps', format='eps')

plt.clf()
fig = plt.figure()
ax = fig.add_subplot(111)
im = ax.scatter(df_both['teff'], df_both['logg'], c=np.sqrt(df_both['teff_var']), vmin=0, vmax=150)
ax.set_xlim(6500,2500)
ax.set_ylim(5.2, 1.5)
#ax.text(6000, 2.0, element_labels[val])
cb = plt.colorbar(im)
cb.set_label('T_eff uncertainty [K]')
plt.savefig('teff_uncer.png', format='png')
plt.savefig('teff_uncer.eps', format='eps')
#
##mkplot(range(14,26))
#plt.savefig('second11.png', format='png')
#plt.savefig('second11.eps', format='eps')
