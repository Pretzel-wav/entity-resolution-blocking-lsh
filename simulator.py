from random import random
import pandas as pd

class SimulationConfig:
    def __init__(self, col:str, generator:callable, likelihood:float):
        if not isinstance(col, str):
            raise ValueError('Column name must be a string.')
        if not callable(generator):
            raise ValueError('Generator must be a callable function.')
        if not isinstance(likelihood, float) or likelihood < 0 or likelihood > 1:
            raise ValueError('Likelihood must be a float between 0 and 1')
        self.col = col
        self.generator = generator
        self.likelihood = likelihood

class Attribute:
    def __init__(self, name, generator, likelihood=1):
        self.name = name
        self.generator = generator
        self.likelihood = likelihood

    def generate(self):
        return self.generator()

def simulate(df:pd.DataFrame, configs:list[SimulationConfig], union:bool=False, tracking_col:str='events') -> pd.DataFrame:
    """
    Run life event simulations on a DataFrame, given the provided configurations.

    Args:
    df (pd.DataFrame): The DataFrame to simulate on.
    configs (list[SimulationConfig]): A list of SimulationConfig objects, each representing a field to simulate.
    union (bool): Whether to return the original DataFrame with the simulated DataFrame concatenated to it. When false, only the simulated DataFrame is returned.
    tracking_col (str): The column name to use for tracking which fields were altered during the simulation.

    Returns:
    pd.DataFrame: The simulated DataFrame.
    """
    if tracking_col == '':
        raise ValueError('Tracking column name cannot be an empty string.')
        
    if tracking_col not in df.columns:
        df.insert(len(df.columns), tracking_col, '')
    output_df = df.copy()

    fields = _generate_fields_from_configs(configs)
    for field in fields:
        output_df = _simulate_field(output_df, field, tracking_col)

    if union:
        return pd.concat([df, output_df], axis=0)
    else:
        return output_df

def _generate_fields_from_configs(configs:list[SimulationConfig]) -> list[Attribute]:
    """
    Generate a list of Attribute objects from a list of SimulationConfig objects.

    Args:
    configs (list[SimulationConfig]): A list of SimulationConfig objects.

    Returns:
    list[Attribute]: A list of Attribute objects.
    """
    fields = []
    for config in configs:
        field = Attribute(name=config.col, generator=config.generator, likelihood=config.likelihood)
        fields.append(field)
    return fields

def _simulate_field(df:pd.DataFrame, field:Attribute, tracking_col:str) -> pd.DataFrame:
    """
    Simulate a life event on a DataFrame column.

    Args:
    df (pd.DataFrame): The DataFrame to simulate on.
    field (Attribute): The field to simulate.
    tracking_col (str): The column name to use for tracking which fields were altered during the simulation

    Returns:
    pd.DataFrame: The simulated DataFrame.
    """
    for index, row in df.iterrows():
        if random() < field.likelihood:
            df.at[index, field.name] = field.generate()
            df.at[index, tracking_col] += ',' + field.name
    return df