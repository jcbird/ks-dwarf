import numpy as np
from astropy.io import fits as pyfits
import pandas as pd
from matplotlib import pyplot as plt
# import matplotlib

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
data_dict['logg'] = np.array(data['FPARAM'][:, 1], dtype=float)
data_dict['teff'] = np.array(data['FPARAM'][:, 0], dtype=float)
for i, label in enumerate(element_labels):
    data_dict[label] = np.array(data['FELEM'][:, i], dtype=float)

df_data = pd.DataFrame(data_dict)
df_m67 = df_data[df_data['field'].str.match('M67')]
df_ple = df_data[df_data['field'].str.match('Pleiades')]

mean_elem_m67 = np.array([df_m67[label].mean() for label in element_labels])
mean_elem_ple = np.array([df_ple[label].mean() for label in element_labels])

#  calculate residuals and make new columns
for i, label in enumerate(element_labels):
    resid_m67 = df_m67[label] - mean_elem_m67[i]
    resid_ple = df_ple[label] - mean_elem_ple[i]
    df_m67['res_'+label] = pd.Series(resid_m67, index=df_m67.index)
    df_ple['res_'+label] = pd.Series(resid_ple, index=df_ple.index)


grp_m67 = df_m67.groupby([pd.cut(df_m67['teff'], bins_teff),
                          pd.cut(df_m67['logg'], bins_logg)])

grp_ple = df_ple.groupby([pd.cut(df_ple['teff'], bins_teff),
                          pd.cut(df_ple['logg'], bins_logg)])


df_both = pd.concat((df_m67, df_ple))
grp_both = df_both.groupby([pd.cut(df_both['teff'], bins_teff),
                            pd.cut(df_both['logg'], bins_logg)])

print "HEY HYE"
uncer = {}
for ii in element_labels:
    uncer[ii] = []

plot_elems = ['[C/H]',  '[O/H]',  '[N/H]',  '[Ti/H]', '[V/H]',
              '[Mg/H]', '[Al/H]']


def mkplot(elem_list):
    with plt.style.context(['ggplot', 'avrfig']):
        fig1, axs1 = plt.subplots(2, 4, sharex=True, sharey=True,
                                  figsize=(12, 7.416))
        fig1.subplots_adjust(wspace=0.001, hspace=0.001)
        for ii, elem in enumerate(elem_list):
            teff = []
            logg = []
            uncer = []
            for name, group in grp_both:
                teff.extend(group['teff'])
                logg.extend(group['logg'])
                res_uncer = group['res_'+elem].std()
                uncer.extend([res_uncer]*len(group['teff']))
            # nanmask = np.isnan(uncer)
            aax = np.ravel(axs1)
            im = aax[ii].scatter(teff, logg, c=uncer, vmin=0, vmax=0.4)
            aax[ii].set_xlim(6700, 2700)
            aax[ii].set_ylim(5.2, 1.8)
            aax[ii].text(6000, 2.0, elem)
            print np.nanmin(uncer), np.nanmax(uncer)
            if ii in [0, 1, 2, 3]:
                aax[ii].set_xticklabels(visible=False)
            if ii in [1, 2, 3, 5, 6]:
                aax[ii].set_yticklabels(visible=False)
        plt.tight_layout()
        fig1.colorbar(im, ax=axs1.ravel().tolist())

mkplot(plot_elems)
# plt.savefig('first14.png', format='png')
# plt.savefig('first14.eps', format='eps')
# plt.clf()
plt.show()
