import pandas as pd

class CO2EmissionAnalyzer:
    def __init__(self, file_path):
        self.df = pd.read_csv(file_path)
        self._prepare_data()
    
    def _prepare_data(self):
        self.df.drop_duplicates(inplace=True)
        self.df.dropna(inplace=True)
        categorical_columns = ['Make', 'Model', 'Vehicle Class', 'Transmission', 'Fuel Type']
        for col in categorical_columns:
            self.df[col] = self.df[col].astype('category')
    
    def get_summary(self):
        return {
            "num_measurements": self.df.shape[0],
            "data_types": self.df.dtypes,
            "missing_values": self.df.isnull().sum(),
            "duplicate_values": self.df.duplicated().sum()
        }
    
    def get_top_fuel_consumers(self, n=3):
        return {
            "highest_city_fuel": self.df.nlargest(n, 'Fuel Consumption City (L/100km)')[['Make', 'Model', 'Fuel Consumption City (L/100km)']],
            "lowest_city_fuel": self.df.nsmallest(n, 'Fuel Consumption City (L/100km)')[['Make', 'Model', 'Fuel Consumption City (L/100km)']]
        }
    
    def get_engine_size_range_data(self, min_size=2.5, max_size=3.5):
        filtered = self.df[(self.df['Engine Size (L)'] >= min_size) & (self.df['Engine Size (L)'] <= max_size)]
        return {
            "num_vehicles": len(filtered),
            "avg_co2_emissions": filtered['CO2 Emissions (g/km)'].mean()
        }
    
    def get_audi_emissions(self):
        audi_cars = self.df[self.df['Make'] == 'Audi']
        audi_4_cyl = audi_cars[audi_cars['Cylinders'] == 4]
        return {
            "num_audi": len(audi_cars),
            "avg_co2_emissions_4cyl": audi_4_cyl['CO2 Emissions (g/km)'].mean()
        }
    
    def get_cylinders_emissions(self):
        return self.df[self.df['Cylinders'].isin([4, 6, 8])].groupby('Cylinders')['CO2 Emissions (g/km)'].mean()
    
    def get_fuel_consumption_by_type(self):
        fuel_stats = {}
        for fuel in ['D', 'X']:  
            fuel_df = self.df[self.df['Fuel Type'] == fuel]
            fuel_stats[fuel] = {
                "avg_city_fuel": fuel_df['Fuel Consumption City (L/100km)'].mean(),
                "median_city_fuel": fuel_df['Fuel Consumption City (L/100km)'].median()
            }
        return fuel_stats
    
    def get_worst_4cyl_diesel(self):
        diesel_4cyl = self.df[(self.df['Cylinders'] == 4) & (self.df['Fuel Type'] == 'D')]
        return diesel_4cyl.nlargest(1, 'Fuel Consumption City (L/100km)')[['Make', 'Model', 'Fuel Consumption City (L/100km)']]
    
    def get_manual_transmission_count(self):
        return len(self.df[self.df['Transmission'].str.startswith('M')])
    
    def get_correlation_matrix(self):
        return self.df.select_dtypes(include=['float64', 'int64']).corr()

if __name__ == "__main__":
    analyzer = CO2EmissionAnalyzer('LV3/resources/data_C02_emission.csv')
    print(analyzer.get_summary())
    print(analyzer.get_top_fuel_consumers())
    print(analyzer.get_engine_size_range_data())
    print(analyzer.get_audi_emissions())
    print(analyzer.get_cylinders_emissions())
    print(analyzer.get_fuel_consumption_by_type())
    print(analyzer.get_worst_4cyl_diesel())
    print(analyzer.get_manual_transmission_count())
    print(analyzer.get_correlation_matrix())
