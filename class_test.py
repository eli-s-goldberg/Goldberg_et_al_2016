import pandas as pd
import math

'''
This importable class converts physicochemical parameters to dimensionless parameters and other needed parameters. There
there is little in the way of error coding built in. As such, if you cannot convert, it is likely that you have mis-
spelled the column header, or do not have the neccessary column data. The formulas for calculating the dimensionless
and other numbers are included in the wiki.

Also note. I do NOT do a units check. The units of the input data must be in meters, kg, seconds, mV, Moles/L. More
detail will be given in the wiki, however. Also, references for formula are embedded in the definitions.

Author: Eli Goldberg February, 2016

'''

try:
    unicode = unicode
except NameError:
    # 'unicode' is undefined, must be Python 3
    str = str
    unicode = str
    bytes = bytes
    basestring = (str, bytes)
else:
    # 'unicode' exists, must be Python 2
    str = str
    unicode = unicode
    bytes = str
    basestring = basestring

# TODO(PandaStabber) This file and class don't get imported or used anywhere.
class pc2dmpc:
    '''
    notes
    -----

    :parameter
    dataframe :
    temp_k :
    diameter_particle :
    diameter_grain :
    darcy_velocity :
    hamaker_constant :
    calculate_kin_visc_by_temp :
    kinematic_water_viscosity :
    solution_pH :
    media_porosity :
    electrolyte_valence :
    column_length :
    column_inner_diameter :

    :return
    '''

    # enm_permittivity_tuple = [
    #     ('C60', 4.4),
    #     ('TiO2', 110),
    #     ('ZnO', 2),
    #     ('CuO', 18.1),
    #     ('MWCNTs', 1328),
    #     ('Ag', 2.65),
    #     ('CeO2', 26),
    #     ('Iron Oxide', 14.2),
    #     ('nHAP', 15.4),
    #     ('nBiochar', 2.9),
    #     ('QDs', 10)
    # ]



    def __init__(self,
                 dataframe,
                 enm_identity='enm_identity',
                 temp_k='tempK',
                 diameter_particle='diameter_particle',
                 diameter_grain='diameter_grain',
                 darcy_velocity='darcy_velocity',
                 hamaker_constant='hamaker_constant',
                 calculate_kin_visc_by_temp=False,  # default value is false
                 kinematic_water_viscosity='kinematic_water_viscosity',
                 solution_pH='solution_pH',
                 media_porosity='media_porosity',
                 particle_density='particle_density',
                 column_length='column_length',
                 column_inner_diameter='column_inner_diameter',
                 influent_pore_volumes='influent_pore_volumes',
                 particle_zeta_potential='particle_zeta_potential',
                 grain_zeta_potential='grain_zeta_potential',
                 enm_permittivity_tuple=[('C60', 4.4)],
                 electrolyte_identity='electrolyte_identity',
                 electrolyte_concentration='electrolyte_concentration',
                 electrolyte_name_valence_tuple=[('NaCl', 1, 1)],
                 electrolyte_ion_1_concentration='electrolyte_ion_1_concentration',
                 electrolyte_ion_2_concentration='electrolyte_ion_2_concentration'
                 ):

        self.dataframe = dataframe
        self.enm_identity = enm_identity
        self.temp_k = temp_k
        self.diameter_particle = diameter_particle
        self.diameter_grain = diameter_grain
        self.darcy_velocity = darcy_velocity
        self.hamaker_constant = hamaker_constant
        self.calculate_kin_visc_by_temp = calculate_kin_visc_by_temp
        self.kinematic_water_viscosity = kinematic_water_viscosity
        self.solution_pH = solution_pH
        self.media_porosity = media_porosity
        self.particle_density = particle_density
        self.column_length = column_length
        self.column_inner_diameter = column_inner_diameter
        self.influent_pore_volumes = influent_pore_volumes
        self.particle_zeta_potential = particle_zeta_potential
        self.grain_zeta_potential = grain_zeta_potential

        assert not isinstance(enm_permittivity_tuple, basestring)
        self.enm_permittivity_tuple = enm_permittivity_tuple

        self.electrolyte_identity = electrolyte_identity
        self.electrolyte_concentration = electrolyte_concentration

        assert not isinstance(electrolyte_name_valence_tuple, basestring)
        self.electrolyte_name_valence_tuple = electrolyte_name_valence_tuple

        self.electrolyte_ion_1_concentration = electrolyte_ion_1_concentration
        self.electrolyte_ion_2_concentration = electrolyte_ion_2_concentration

    def get_aspect_ratio(self):
        '''
        notes
        -----

        :parameter
        :return:
        '''

        dataframe = self.dataframe

        def aspect_ratio_apply(row):
            diam_part = row[self.diameter_particle]
            grain_diam = row[self.diameter_grain]
            return float(diam_part / grain_diam)

        return dataframe.apply(aspect_ratio_apply, 1)

    def get_peclet_number(self):
        '''
        parameters
        ----------
        :return:
        '''

        dataframe = self.dataframe

        def dim_peclet_num_assign(row):
            boltzmann_constant = float(1.3806504 * 10 ** -23)
            temp_K = row[self.temp_k]
            diam_part = row[self.diameter_particle]
            grain_diam = row[self.diameter_grain]
            darcy_vel = row[self.darcy_velocity]
            kin_visc_water = row[self.kinematic_water_viscosity]
            stokes_einstein_diffusion_coefficient = float(boltzmann_constant * temp_K /
                                                          (3 * math.pi * kin_visc_water * diam_part))
            return float(darcy_vel * grain_diam / stokes_einstein_diffusion_coefficient)

        return dataframe.apply(dim_peclet_num_assign, 1)

    def get_gravity_number(self):
        '''
        parameters
        ----------
        :return:
        '''

        dataframe = self.dataframe

        if self.calculate_kin_visc_by_temp == False:
            def gravity_number(row):
                density_water = 1000
                temp_K = row[self.temp_k]
                darcy_vel = row[self.darcy_velocity]
                radius_part = row[self.diameter_particle] / 2
                kin_visc = row[self.kinematic_water_viscosity]
                density_part = row[self.particle_density]

                numerator = (2 * radius_part ** 2) * (density_part - density_water) * 9.81
                demoninator = 8 * kin_visc * darcy_vel
                return float(numerator / demoninator)

        else:

            def gravity_number(row):
                A = float(-3.7188)
                B = float(578.919)
                C = float(-137.546)
                density_water = 1000
                temp_K = row[self.temp_k]
                darcy_vel = row[self.darcy_velocity]
                radius_part = row[self.diameter_particle] / 2
                kin_visc = float(math.exp(A + (B / (C + temp_K))))
                density_part = row[self.particle_density]

                numerator = (2 * radius_part ** 2) * (density_part - density_water) * 9.81
                demoninator = 8 * kin_visc * darcy_vel
                return float(numerator / demoninator)

        return dataframe.apply(gravity_number, 1)

    def get_attractive_number(self):
        '''
        notes
        -----

        :parameter
        :return:
        '''

        dataframe = self.dataframe

        def attraction_number(row):
            radius_part = row[self.diameter_particle] / 2
            darcy_vel = row[self.darcy_velocity]
            ham_constant = row[self.hamaker_constant]

            denominator = 12 * math.pi * radius_part ** 2 * darcy_vel
            return float(darcy_vel / denominator)

        return dataframe.apply(attraction_number, 1)

    def get_particle_pumped_mass(self):
        '''
        notes
        -----

        :parameter


        :return:
        '''

        dataframe = self.dataframe

        def mass_flow(row):
            col_len = row[self.column_length]
            col_id = row[self.column_inner_diameter]
            med_porosity = row[self.media_porosity]
            inf_pore_vols = row[self.influent_pore_volumes]

            col_xsn_area = float(math.pi / 4 * col_id ** 2)

            return float(col_len * col_id * col_xsn_area * med_porosity * inf_pore_vols)

        return dataframe.apply(mass_flow, 1)

    def get_electrokinetic_1(self):
        '''
        notes
        -----

        :parameter

        :return:
        '''

        dataframe = self.dataframe

        def electrokinetic1(row):
            dialectric_constant_water = 7.83e-9  # dialectric constant of water at 25C in coulombs/volt/m
            radius_part = row[self.diameter_particle] / 2
            boltzmann_constant = float(1.3806504 * 10 ** -23)
            temp_K = row[self.temp_k]
            part_zeta = row[self.particle_zeta_potential] / 1e3
            grain_zeta = row[self.grain_zeta_potential] / 1e3

            return float(dialectric_constant_water * radius_part * (part_zeta ** 2 + grain_zeta ** 2) /
                         (4 * boltzmann_constant * temp_K))

        return dataframe.apply(electrokinetic1, 1)

    def get_electrokinetic_2(self):
        '''
        notes:
        -----

        :parameter
        :return:
        '''
        dataframe = self.dataframe

        def electrokinetic2(row):
            part_zeta = row[self.particle_zeta_potential] / 1e3
            grain_zeta = row[self.grain_zeta_potential] / 1e3

            numerator = 2 * (part_zeta / grain_zeta)
            denominator = 1 + (part_zeta / grain_zeta) ** 2
            return float(numerator / denominator)

        return dataframe.apply(electrokinetic2, 1)

    def get_relative_permittivity(self):
        '''
        notes
        -----
        Passing a tuple of particle id'd and their respective relative permittivity values. Yes, they will be perfectly
        correlated...however, the permittivity value is used in other calculations, so it is critical to actualy input
        a number

        :parameter

        :return: return a dataframe of relative permittivity values for a defined set of enm ids within a dataframe
        '''

        dataframe = self.dataframe

        enm_permittivity_tuple = self.enm_permittivity_tuple

        def assign_relative_permit(row):

            # get all the names in the tuples with a list comprehension
            enm_permittivity_tuple_name_list = [x[0] for x in enm_permittivity_tuple]
            enm_permittivity_tuple_value_list = [x[1] for x in enm_permittivity_tuple]

            # then iterate through list and assign permittivity values
            tmp_relative_permit = []
            for enm_tup in zip(enm_permittivity_tuple_name_list, enm_permittivity_tuple_value_list):
                if row[self.enm_identity] == enm_tup[0]:
                    tmp_relative_permit = enm_tup[1]
            return tmp_relative_permit

        return dataframe.apply(assign_relative_permit, 1)

    def get_london_force(self):
        '''
        notes
        -----

        :parameter
        :return:
        '''
        dataframe = self.dataframe

        def london_force(row):
            boltzmann_constant = float(1.3806504 * 10 ** -23)
            temp_K = row[self.temp_k]
            ham_constant = row[self.hamaker_constant]

            return float(ham_constant / (6 * boltzmann_constant * temp_K))

        return dataframe.apply(london_force, 1)

    def get_happel_porosity_parameter(self):
        '''
        notes
        -----
        :parameter

        :return:
        '''

        dataframe = self.dataframe

        def porosity_happel(row):
            med_porosity = row[self.media_porosity]

            gam = (1 - med_porosity) ** (.333333333)
            numerator = 2 * (1 - gam ** 5)
            denominator = 2 - 3 * gam + 3 * gam ** 5 - 2 * gam ** 6
            return float(numerator / denominator)

        return dataframe.apply(porosity_happel, 1)

    def get_ionic_strength(self):

        dataframe = self.dataframe
        electrolyte_name_valence_tuple = self.electrolyte_name_valence_tuple

        def assign_ionic_strength(row):
            electrolyte_name_valence_tuple_name_list = [x[0] for x in electrolyte_name_valence_tuple]
            electrolyte_name_valence_tuple_valence_1_list = [x[1] for x in electrolyte_name_valence_tuple]
            electrolyte_name_valence_tuple_valence_2_list = [x[2] for x in electrolyte_name_valence_tuple]

            ionic_strength = []
            for electrolyte_tup in zip(electrolyte_name_valence_tuple_name_list,
                                       electrolyte_name_valence_tuple_valence_1_list,
                                       electrolyte_name_valence_tuple_valence_2_list):

                if row[self.electrolyte_identity] == electrolyte_tup[0]:

                    if row['electrolyte_identity'] != 'none' and row['electrolyte_concentration'] > 0:
                        ion_1_concentration = row[self.electrolyte_concentration] * row[
                            self.electrolyte_ion_1_concentration]
                        ion_2_concentration = row[self.electrolyte_concentration] * row[
                            self.electrolyte_ion_2_concentration]
                        valence_1 = electrolyte_tup[1]
                        valence_2 = electrolyte_tup[2]
                        ionic_strength = 0.5 * (
                        (ion_1_concentration * valence_1 ** 2) + (ion_2_concentration * valence_2 ** 2))
                        return ionic_strength

                    elif row['electrolyte_identity'] == 'none' and row['electrolyte_concentration'] > 0:
                        print 'No electrolyte identity entered, but non-pH based electrolyte concentration of', \
                            row['electrolyte_concentration'], 'detected for dataframe entry: '
                        print row

                    else:
                        '''
                        if the electrolyte concentration is zero or there is no electrolyte (e.g., DI water), determine the
                        determine the ionic strength using the pH.
                        '''

                        h_plus = float(1 * 10 ** (float(row[self.solution_pH]) - 14))
                        h_plus_valence = 1
                        oh_minus = float(1 * 10 ** (-1 * float(row[self.solution_pH])))
                        oh_minus_valence = 1
                        ionic_strength = 0.5 * ((h_plus * h_plus_valence ** 2) + (oh_minus * oh_minus_valence ** 2))
                        return ionic_strength
                else:
                    print 'Electrolyte identity not found in electrolyte identity tuple... keep it together.'

        return dataframe.apply(assign_ionic_strength,1)

def main():
    enm_permittivity_tuple = [
        ('C60', 4.4),
        ('TiO2', 110),
        ('ZnO', 2),
        ('CuO', 18.1),
        ('MWCNTs', 1328),
        ('Ag', 2.65),
        ('CeO2', 26),
        ('Iron Oxide', 14.2),
        ('nHAP', 15.4),
        ('nBiochar', 2.9),
        ('QDs', 10)
    ]

    electrolyte_name_valence_tuple = [
        ('NaCl', 1, 1),
        ('CaCl2', 2, 1),
        ('KCl', 1, 1),
        ('KNO3', 1, 1),
        ('NaHCO3', 1, 1)
    ]

    df = pd.DataFrame()
    df['tempK'] = [290, 290]
    df['diameter_particle'] = [100e-9, 50e-9]
    df['diameter_grain'] = [100e-9, 500e-6]
    df['darcy_velocity'] = [1e-5, 1e-6]
    df['hamaker_constant'] = [1e-20, 2.7e-21]
    df['kinematic_water_viscosity'] = [1e-4, 1e-4]
    df['solution_pH'] = [6, 7]
    df['media_porosity'] = [0.36, 0.29]
    df['particle_density'] = [1e5, 2650]

    a = pc2dmpc(
        dataframe=df,
        temp_k='tempK',
        diameter_particle='diameter_particle',
        diameter_grain='diameter_grain',
        darcy_velocity='darcy_velocity',
        hamaker_constant='hamaker_constant',
        solution_pH='solution_pH',
        media_porosity='media_porosity',
        enm_permittivity_tuple=enm_permittivity_tuple
    )


if __name__ == "__main__":
    main()
