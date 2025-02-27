import numpy as np 
from numba import njit, float64
from numba.experimental import jitclass

USD_to_AUD = 1.43 # AUD to USD where necessary
discount_rate = 0.0599 # Real discount rate - same as gencost
USD_inflation = 1.18 # 2020->2023
AUD_inflation = 1.16 # 2020->2023
MWh_per_GJ = 0.27778
carbon_price = 0 # AUD/t
tCO2e_per_GJ_gas  = 0.05 
tCO2e_per_GJ_coal = 0.1


## costs come from Apx Table B.9 of GenCost 2023-24 
## year = 2023
#==============================================================================
# utility solar
csiro_pv = (
    1526,   # capex AUD/kW 
    17,     # fom   AUD/kW p.a.
    0,      # vom   AUD/MWh 
    30,     # life  years
    )

# onshore wind
csiro_onsw = (
    3038,   # capex AUD/kW 
    25,     # fom   AUD/kW p.a.
    0,      # vom   AUD/MWh 
    25,     # life  years
    )

# offshore wind
csiro_offw = (
    5545,   # capex AUD/kW 
    149.9,  # fom   AUD/kW
    0,      # vom   AUD/MWh
    25,     # life  years
    )

# large open cycle gas 
csiro_gas = (
    943,    # capex AUD/kW
    10.2,   # fom   AUD/kW p.a.
    7.3 +   # vom   AUD/MWh
    16.5 / 0.33 * MWh_per_GJ + # fuel AUD/MWh
    tCO2e_per_GJ_gas / MWh_per_GJ * carbon_price, # carbon price AUD/MWh
    25,     # life  years
    )

# black coal
csiro_coal = (
    5616,   # capex AUD/kW
    53.2,   # fom   AUD/kW p.a.
    4.2 +   # vom   AUD/MWh
    7.8 / 0.42 * MWh_per_GJ + # fuel AUD/MWh
    tCO2e_per_GJ_coal / MWh_per_GJ * carbon_price, # carbon price AUD/MWh
    30,     # life  years
    )

## costs adjusted for inflation but otherwise unchanged from Lu et al. 2021 https://doi.org/10.1016/j.energy.2020.119678
#==============================================================================
hvdc_overhead = (
    320 * AUD_inflation,  # capex AUD/MW-km
    3.2 * AUD_inflation,  # fom   AUD/MW-km p.a.
    0,                    # vom   AUD/MWh
    50,                   # life  years
    )

converter = (
    160 * AUD_inflation,  # capex AUD/kW
    1.6 * AUD_inflation,  # fom   AUD/kW p.a.
    0,                    # vom   AUD/MWh
    30,                   # life  years
    )

# undersea costs includer converter
hvdc_undersea = (
    4000 * AUD_inflation, # capex AUD/MW-km
    40   * AUD_inflation, # fom   AUD/MW-km p.a.
    0,                    # vom   AUD/MWh
    30,                   # life  years
    )

hvac = (
    1500 * AUD_inflation, # capex AUD/MW-km
    15   * AUD_inflation, # fom   AUD/MW-km p.a.
    0,                    # vom   AUD/MWh
    50,                   # life  years
    )

## costs from re100 cost model - Class A site
#==============================================================================
phes = (
    530    * USD_inflation * USD_to_AUD, # capex AUD/kW
    47     * USD_inflation * USD_to_AUD, # capex AUD/kWh
    8.21   * USD_inflation * USD_to_AUD, # fom AUD/kW p.a.
    0.3    * USD_inflation * USD_to_AUD, # vom AUD/MWh
    112000 * USD_inflation * USD_to_AUD, # AUD per replace
    50,  # replace lifetime
    100, # life years
    )

# same O&M as PHES, but no capital
hydro = (
    0, # capex (existing only)
    8.21   * USD_inflation * USD_to_AUD, # fom AUD/kW p.a. #same as phes (approx)
    0.3    * USD_inflation * USD_to_AUD, # vom AUD/MWh #same as phes (approx)
    50, #life years
    )


@njit
def annualization_constants(capex, fom, vom, life, dr):
    """ 
    Calculate annualized costs parametrically for power and energy 
    Input:
        capex - $/kW
        fom   - $/kW p.a.
        vom   - $/MWh
        life  - years
        dr    - %
    Output:
        discounted capex cost factor ($ p.a. / GW)
        fom cost factor ($ p.a. / GW)
        vom cost factor ($ p.a. / MWh p.a.)
    """
    pv = (1-(1+dr)**(-1*life))/dr
    return np.array([1_000_000 * capex / pv, # $ p.a./GW
                     1_000_000 * fom, # $ p.a./GW
                     1000 * vom, # $ p.a./GWh p.a.
                     ], np.float64)

@njit
def annualization_transmission_constants(capex, fom, vom, life, d, dr):
    """ 
    Calculate annualized costs parametrically for power and energy, for transmission lines only
    Input:
        capex - $/MW-km
        fom   - $/MW-km p.a.
        vom   - $/MWh
        life  - years
        d     - km
        dr    - %
    Output:
        discounted capex cost factor ($ p.a. / GW)
        fom cost factor ($ p.a. / GW)
        vom cost factor ($ p.a. / MWh p.a.)
    """
    pv = (1-(1+dr)**(-1*life))/dr
    return np.array([d * capex * 1000 / pv,# $ p.a./GW
                     d * fom * 1000, # $ p.a./GW
                     vom * 1000, # $ p.a./GWh p.a.
                     ])

@njit
def annualization_phes_constants(capex_p, capex_e, fom, vom, replace_cost, replace_life, life, dr):
    """ Calculate annualized costs parametrically for power and energy, for PHES only 
    capex_p, fom: AUD/kW
    capex_e: AUD/kWh
    vom: AUD/MWh
    replace: AUD per replace
    replace_life: years """
        
    pv = (1-(1+dr)**(-1*life))/dr
    
    return np.array([
            capex_p * 1_000_000 / pv, # capex $ p.a./GW
            capex_e * 1_000_000 / pv, # capex $ p.a./GWh
            fom * 1_000_000, # fom $ p.a./GW            
            vom, # vom $ p.a./MWh p.a.
            replace_cost * ((1+dr)**(-1*replace_cost) + (1+dr)**(-1*replace_life*2)) / pv # replace capex $p.a.
            ])


@jitclass([
    ('pv',      float64[:] ),  
    ('onsw',    float64[:] ),  
    ('offw',    float64[:] ),
    ('gas',     float64[:] ),
    ('coal',    float64[:] ),
    ('hydro',   float64[:] ),
    ('phes',    float64[:] ),
    ('ac',      float64[:] ),
    ('hvdc',    float64[:] ),
    ])
class cost_factors:
    def __init__(self, scenario, DClengths=np.array([], np.int64), undersea_mask=np.array([], np.bool_), network_mask=np.array([], np.bool_)):
        self.pv   = annualization_constants(*csiro_pv,   discount_rate)
        self.onsw = annualization_constants(*csiro_onsw, discount_rate)
        self.offw = annualization_constants(*csiro_offw, discount_rate)
        
        self.gas  = annualization_constants(*csiro_gas,  discount_rate)
        self.coal = annualization_constants(*csiro_coal, discount_rate)

        self.hydro = annualization_constants(*hydro, discount_rate)
        self.phes = annualization_phes_constants(*phes, discount_rate)

        self.ac   = annualization_transmission_constants(*hvac, 20, discount_rate)
        self.hvdc = np.zeros(len(network_mask), np.float64)
        if scenario >= 21:
            for i, undersea in enumerate(undersea_mask):
                if network_mask[i] is False:
                    continue
                if undersea:
                    self.hvdc[i] = annualization_transmission_constants(*hvdc_undersea, DClengths[i], discount_rate)[0] # vom is 0
                else: 
                    self.hvdc[i] = annualization_transmission_constants(*hvdc_overhead, DClengths[i], discount_rate)[0] # vom is 0
                    self.hvdc[i] += 2*annualization_constants(*converter, discount_rate)[0]


if __name__ == '__main__':
    from Input import scenario, DClengths, undersea_mask, network_mask
    
    costs = cost_factors(scenario, DClengths, undersea_mask, network_mask)