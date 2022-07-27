**Summary of 4C analysis**

_Matthew Menary, Paris le 27/7/2022_

This document summaries the progress I have made working in the 4C project, as well as where I think future analysis could go. It also provided links to the code and data.

All the code and this document can be found here:


```
https://github.com/mattofficeuk/4C_fgco2 
```


Data stored at: <code> /thredds/ipsl/mmenary/fgco2/</code></strong>

**Abstract:**

In order to improve our ability to predict the near-term evolution of climate, it may be important to accurately predict the evolution of atmospheric CO2, and thus carbon sinks. Following on from process-driven improvements of decadal predictions in physical oceanography, we focus on improving our understanding of the internal processes and variables driving CO2 uptake by the North Atlantic ocean. Specifically, we use the CMIP6 model IPSLCM6A to investigate the drivers of ocean-atmosphere CO2 flux variability in the North Atlantic subpolar gyre (NA SPG) on seasonal to decadal timescales. We find that DpCO2 (CO2 partial pressure difference between atmosphere and ocean) variability dominates over sea surface temperature (SST) and sea surface salinity (SSS) variability on all timescales within the NA SPG. Meanwhile, at the ice-edge, there are significant roles for both ice concentration and surface winds in driving the overall CO2 flux changes. Investigating the interannual DpCO2 variability further, we find that this variability is itself driven largely by variability in simulated mixed layer depths in the northern SPG. On the other hand, SSTs show an important contribution to DpCO2 variability in the southern SPG and on longer (decadal) timescales. Initial extensions into a multi-model context show similar results. By determining the key regions and processes important for skilful decadal predictions of ocean-atmosphere CO2 fluxes, we aim to both improve confidence in these predictions as well as highlight key targets for climate model improvement. 

**Short summary of results:**



1. Scientific question: “What physical variables do we need to be able to reliably predict - if we want to make annual to decadal predictions of the air sea CO2 flux?”
    1. In the North Atlantic, it is dpCO2.
2. Follow up question 1: What drives dpCO2 variability, and is it timescale dependent?
    2. It is mostly SST and mixed layer depths (MLD) in the North Atlantic subpolar gyre. Summer seems to be the important season. On annual timescales MLDs are most important, whereas on decadal timescales it is SSTs in the Labrador Sea.
3. Follow up question 2: Are these variables (SSTs in the Labrador Sea; MLDs in the SPG) predictable in current generation annual-to-decadal prediction systems:
    3. Analysis ongoing, but it looks like “yes”

**Code:**

Notebooks


<table>
  <tr>
   <td><strong>File name</strong>
   </td>
   <td><strong>Description</strong>
   </td>
  </tr>
  <tr>
   <td><code>CO2_recalculations</code>
   </td>
   <td>Cosme’s original code, kept for reference
   </td>
  </tr>
  <tr>
   <td><code>CO2_Offline_Correlations</code>
   </td>
   <td>Breaking down the fgco2 equation as a function of timescale. This code is adapted from Cosme's original code. The first plots demonstrate if/when we can get the offline FGCO2 and online FGCO2 to match adequately. The subsequent plots then look at the effect of 1) holding one variable constant, or 2) holding all variables EXCEPT one constant: Basically two different ways of estimating the importance of different variables. CONCLUSION: DPCO2 is very important on all timescales in the North Atlantic SPG.
<p>
See below for a batch script version of this.
   </td>
  </tr>
  <tr>
   <td><code>CO2_Offline_TimeSeries</code>
   </td>
   <td>To make some time series and then investigate the effect of TIMESCALE on the data plotted in CO2_Offline_Correlations
   </td>
  </tr>
  <tr>
   <td><code>dpco2_MLR_PlotMaps</code>
   </td>
   <td>This notebook calculates the correlation between the OFFLINE DPCO2 Multiple Linear Regression (MLR) and the online calculation of DPCO2. It allows us to then HOLD ONE VARIABLE CONSTANT and estimate the effect/importance of that variable
   </td>
  </tr>
  <tr>
   <td><code>dpco2_MLR_ProofOfConcept</code>
   </td>
   <td>To check and show that we can do the multiple linear regression (MLR) between DPCO2 and it’s components correctly
   </td>
  </tr>
  <tr>
   <td><code>getNorthAtlanticBlocks</code>
   </td>
   <td>Utility code to extract the North Atlantic region. This speeds things up as, in order to avoid issues with file sizes, I break the global domain up into “blocks” when doing the batch processing
   </td>
  </tr>
  <tr>
   <td><code>QuickCompareHistPiCon</code>
   </td>
   <td>To check that different methods of calculating fgco2 online/offline give comparable results
   </td>
  </tr>
  <tr>
   <td><code>Compare_Wind_Tau</code>
   </td>
   <td>To check and see if using tauuo/vo (which are on the ocean grid) is okay for estimating wind speed (which is on the atmosphere grid). Answer: Not really...
   </td>
  </tr>
</table>


(Batch) scripts:


<table>
  <tr>
   <td><code>SUBMIT_*sh</code>
   </td>
   <td>Submits the relevant script
   </td>
  </tr>
  <tr>
   <td><code>wrapper_*</code>
   </td>
   <td>An intermediate script for the submission process
   </td>
  </tr>
  <tr>
   <td><code>dpco2_MLR.py</code>
   </td>
   <td>Batch script to compute the dpco2 MLR
   </td>
  </tr>
  <tr>
   <td><code>dpco2_MLR_PlotMaps.py</code>
   </td>
   <td>Batch script to make the dpco2 MLR maps for all models
   </td>
  </tr>
  <tr>
   <td><code>dpco2_MLR_PlotMaps_ByModel.py</code>
   </td>
   <td>As above but for single models
   </td>
  </tr>
  <tr>
   <td><code>CO2_offline.py</code>
   </td>
   <td>Batch script to compute fgco2 offline, and then to look at the effect of holding individual components constant/letting other vary. Creates time series’
   </td>
  </tr>
  <tr>
   <td><code>CO2_offline_corrs.py</code>
   </td>
   <td>As above but creates maps
   </td>
  </tr>
</table>

