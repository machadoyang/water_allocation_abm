# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

from mesa import Agent, Model
from mesa.space import NetworkGrid
import DataPreparation

import numpy as np
import pandas as pd
import scipy.stats as ss
import random

from collections import defaultdict

# class WaterUser(Agent):
#     _last_id = 0
#     def __init__(self, model):
#         super().__init__(WaterUser._last_id+1, model)
#         WaterUser._last_id += 1

#     def step(self):
#         pass


class FarmerAgent(Agent):
    _last_id = 0

    def __init__(self, model):
        super().__init__(FarmerAgent._last_id+1, model)
        FarmerAgent._last_id += 1
        self.type = 'farmer'

        """MNL parameters
        Attention to distributions!
        """
        self.b_fruits = 0 # reference for dummy variable
        self.b_vegetables = np.random.normal(
            model.farmer_parameters.loc['b_vegetables']['mean'],
            model.farmer_parameters.loc['b_vegetables']['sd'])
        self.b_maize_cassava_beans = np.random.normal(
            model.farmer_parameters.loc['b_maize_cassava_beans']['mean'],
            model.farmer_parameters.loc['b_maize_cassava_beans']['sd'])
        self.b_no_binding = np.random.normal(
            model.farmer_parameters.loc['b_no_binding']['mean'],
            model.farmer_parameters.loc['b_no_binding']['sd'])
        self.b_irrigation_efficiency = np.random.normal(
            model.farmer_parameters.loc['b_irrigation_efficiency']['mean'],
            model.farmer_parameters.loc['b_irrigation_efficiency']['sd'])
        self.b_no_technical_assistance = 0 # reference for dummy variable
        self.b_technical_assistance = np.random.normal(
            model.farmer_parameters.loc['b_technical_assistance']['mean'],
            model.farmer_parameters.loc['b_technical_assistance']['sd'])
        self.b_selling_secured = np.random.normal(
            model.farmer_parameters.loc['b_selling_secured']['mean'],
            model.farmer_parameters.loc['b_selling_secured']['sd'])
        self.b_water_price = np.random.lognormal(
            model.farmer_parameters.loc['b_water_price']['mean'],
            model.farmer_parameters.loc['b_water_price']['sd'])
        
        """Farmer init characteristics"""
        self.irrigation_eff = np.random.choice([0.8, 0.9, 0.95], 1)[0]

        """Farmer parameters"""
        self.chosen_contract = None
        self.farm_size = ss.halfnorm.rvs()
        self.chosen_crop = None
        self.amount_of_water_needed = 0
        self.yearly_revenue = 0
        
        self.received_water_right = False
        self.p_to_override = np.random.beta(a=2, b=5, size=1)[0]  # Probability to override
        self.go_rogue = False
        
        
        """Other parameters"""
        self.life_time = 0

    def choose_contract(self):
        # Get manager contract options
        contract_options = model.agents.select(lambda agent:agent.type == "manager").get("contract_options")[0]    
        
        # substitute farmer own irrigationefficiency
        for contract in contract_options:
            contract['irrigation_eff'] = self.irrigation_eff
        
        def calculate_utilities(contract_options):
            utilities = []
            for contract in contract_options:
                utility = (
                self.b_fruits*(contract['crop']=="fruits") +
                self.b_vegetables*(contract['crop']=="vegetables") +
                self.b_maize_cassava_beans*(contract['crop']=="maize_cassava_beans") +
                self.b_no_binding*(contract['crop']=="no_binding") +
                self.b_irrigation_efficiency*contract['irrigation_eff'] + 
                self.b_no_technical_assistance*(contract['tech_assistance']==0) +
                self.b_technical_assistance*(contract['tech_assistance']==1) +
                self.b_selling_secured*contract['sell_secure'] +
                self.b_water_price*contract['water_price']
                )
                utilities.append(utility)
            return utilities
        
        def calculate_probabilities(utilities):
            # Calculate probabilities based on multinomial logit model
            denominator = 0
            probabilities = []
            for utility in utilities:
                denominator += np.exp(utility)
            for utility in utilities:
                prob = np.exp(utility)/denominator
                probabilities.append(prob)
            return probabilities
        
        def decide_contract(probs, contract_options):
            chosen_contract_id = np.random.choice(
                list(range(len(probs))),
                1, p=probs) # array of values, size, probabilities
            chosen_contract = contract_options[chosen_contract_id[0]] # chosen contract returns array
            return chosen_contract
        
        utilities = calculate_utilities(contract_options)
        probabilities = calculate_probabilities(utilities)
        chosen_contract = decide_contract(probabilities, contract_options)
        self.chosen_contract = chosen_contract
        
    def choose_crop(self):
        """
        If farmer choose a contract, they plant whatever is in the contract.
        If they choose status quo, they plant based on market share
        """
        if (self.chosen_contract['crop'] == "no_binding"):
            self.chosen_crop = "fruits"
        else:
            self.chosen_crop = self.chosen_contract['crop']

    def calculate_water_to_withdraw(self):
        # in m³/year
        self.amount_of_water_needed = self.farm_size * model.crop_parameters[self.chosen_crop].loc['irrigation_volume'] * 100 / self.irrigation_eff
        
    def consider_go_rogue(self):
        if (self.p_to_override < model.override_threshold):
            self.go_rogue = True
    
    def ask_for_water(self):
        pass
    
    def produce(self):
        if (self.chosen_crop == "fruits"):
            production_per_ha = 1
        elif (self.chosen_crop == "vegetables"):
            production_per_ha = 2
        else: # it can only be maize_cassava_beans
            production_per_ha = 3
        self.yearly_revenue = self.farm_size * production_per_ha * model.crop_parameters.loc['revenue'][self.chosen_crop]
    
    def step_stage_one(self):
        self.choose_contract()
        self.choose_crop()
        self.calculate_water_to_withdraw()
        if ((self.life_time == 0) and (self.chosen_crop == "no_binding")): # farmer can go rogue before asking water to the manager if they chose to not get a contract
            self.consider_go_rogue()
            
    def step_stage_two(self):
        if ((self.life_time == 0) and (not self.received_water_right)): # farmer consider to go rogue again if water request is denied (regardless of contract)
            self.consider_go_rogue()
        self.produce()
        # farmer produce for the year
        pass


class HumanSupplyAgent(Agent):
    _last_id = 0

    def __init__(self, model):
        super().__init__(HumanSupplyAgent._last_id+1, model)
        HumanSupplyAgent._last_id += 1
        self.type = 'human_supply'
        
        """water right parameters"""
        self.amount_of_water_needed = np.random.normal()
        self.received_water_right = False
        self.p_to_override = np.random.beta(a=2, b=5, size=1)[0]  # Probability to override
        
        """Other parameters"""
        self.life_time = 0

    def step(self):
        pass
        # print("Human supply user at in position {} withdrew {} m³/year".format(
        #     self.pos, self.amount_of_water_needed))


class IndustryAgent(Agent):
    _last_id = 0

    def __init__(self, model):
        super().__init__(IndustryAgent._last_id+1, model)
        IndustryAgent._last_id += 1
        self.type = 'industry'
        
        """water right parameters"""
        self.amount_of_water_needed = np.random.normal()
        self.received_water_right = 0
        self.p_to_override = np.random.beta(a=2, b=5, size=1)[0]  # Probability to override
        
        """Other parameters"""
        self.life_time = 0

    def step(self):
        print("I'm the industry")


class ManagerAgent(Agent):
    def __init__(self, unique_id, model):
        super().__init__(unique_id, model)
        self.type = 'manager'
        self.normal_water_price = 0.38
        self.contract_options = [
            {"crop": "fruits", "irrigation_eff": None, "tech_assistance": False, "sell_secure": 0, "water_price": self.normal_water_price},
            {"crop": "vegetables", "irrigation_eff": None, "tech_assistance": False, "sell_secure": 0, "water_price": self.normal_water_price},
            {"crop": "maize_cassava_beans", "irrigation_eff": None, "tech_assistance": False, "sell_secure": 0, "water_price": self.normal_water_price},
            {"crop": "no_binding", "irrigation_eff": None, "tech_assistance": False, "sell_secure": 0, "water_price": self.normal_water_price}]
        
        """Implement water policies"""
        self.implement_technical_assistance()
        
    def allocate_water(self):
        if model.verbose == True:
            print(
                "Manager is conceiving water rights on the basis of first come first served.")
            print("Priorization order: Human supply > Industrial > Agriculture")
            print("Distributing water")
        agents_contents = self.model.grid.get_all_cell_contents()
        # sort agents by unique_id and their type
        agent_type_order = {"human_supply": 1, "industry":2, "farmer":3}
        agents_contents.sort(key=lambda x: (agent_type_order.get(x.type, float('inf')), x.unique_id))
        for agent in agents_contents:
            if (agent.type == 'farmer' and agent.life_time == 0):
                # get section information where farmer is positioned
                farmer_section = model.G.nodes[agent.pos]["section"]
                if (model.virtual_water_available_per_section[str(farmer_section)] >= agent.amount_of_water_needed):
                    # Conceive water right and deduct from available water per section
                    agent.amount_of_water_received = agent.amount_of_water_needed
                    model.virtual_water_available_per_section[str(
                        farmer_section)] -= agent.amount_of_water_needed
                    agent.received_water_right = True
                    if model.verbose == True:
                        print("Farmer n. " + str(agent.unique_id) + " received " +
                              str(agent.amount_of_water_received) + " m³/year")
                        print("Remaining virtual water available on section " + str(farmer_section) +
                              " is " + str(model.virtual_water_available_per_section[str(farmer_section)]))
                else:
                    if model.verbose == True:
                        print("Water request denied to farmer {}. There is no available water at section {}.". format(
                            agent.unique_id, farmer_section))

    def implement_technical_assistance(self):
        """The manager implement technical assistance and incentive the production of fruits.
        This change farmer contract options and put a discount on the price of water to fruits."""
        for contract in self.contract_options:
            contract['tech_assistance'] = True
            if contract['crop'] == "fruits":
                contract['water_price'] -= 0.05
                
    def change_water_price(self, price):
        """Change contracts water price"""
        for contract in self.contract_options:
            contract['water_price'] = price
        pass
    
    def step(self):
        # print("Hi, I am the manager n. {} and I am in position {}".format(
        #     self.unique_id, self.pos))
        self.allocate_water()


class WaterAllocationModel(Model):
    current_id = 0

    def __init__(
            self,
            linear_graph,
            init_water_available_per_section,
            n_farmers_to_create_per_year,
            farmer_parameters,
            crop_parameters):
        super().__init__()

        # Set model parameters
        self.G = linear_graph
        self.grid = NetworkGrid(self.G)
        self.number_of_farmers_to_create = n_farmers_to_create_per_year
        self.farmer_parameters = farmer_parameters
        self.technical_assistance = False
        self.available_water_per_section = available_water_per_section.copy()
        self.init_water_available_per_section = available_water_per_section.copy()
        self.virtual_water_available_per_section = init_water_available_per_section.copy()
        self.override_threshold = 0.3
        self.crop_parameters = crop_parameters
        # self.init_water_available_per_section = self.allocate_all_water_to_section_one().copy()
        self.verbose = False
        self.time = 0
        
    def withdraw_water(self):
        """
        Withdraws water from the canal upstream to downstream.
        """

        agents_contents = self.grid.get_all_cell_contents()
        grouped_agents = defaultdict(list)
        # Separate agents from each section to account for spatiallity
        for agent in agents_contents:
            grouped_agents[model.G.nodes[agent.pos]["section"]].append(agent)

        # To account for spatiality, we transfer all water from each section to first element of that section in grouped_agents.keys()
        # Once that agent withdraws, we move all water available to next, and so on
        # Algorithm breaks if an agent allocates itself because it checks water from previous node, so we prevent them from do it in random allocation
        current_list_of_sections = list(grouped_agents.keys())
        model.available_water_per_section[str(
            current_list_of_sections[0])] = model.available_water_per_section[str(1)]

        for section in grouped_agents:
            for agent in grouped_agents[section]:
                if (agent.type != 'manager'):
                    # get section information where farmer is positioned
                    farmer_section = model.G.nodes[agent.pos]["section"]
                    if (model.available_water_per_section[str(farmer_section)] >= agent.amount_of_water_needed):
                        if (agent.received_water_right == True):
                            agent.amount_of_water_withdrawn = agent.amount_of_water_received
                            model.available_water_per_section[str(
                                farmer_section)] -= agent.amount_of_water_received
                            if model.verbose == True:
                                print("Farmer {} have withdrawn {} m³/year.". format(
                                    agent.unique_id, agent.amount_of_water_withdrawn))
                        else:
                            if (agent.p_to_override < model.override_threshold):
                                agent.agent_performed_override = True
                                agent.amount_of_water_withdrawn = agent.amount_of_water_needed
                                model.available_water_per_section[str(
                                    farmer_section)] -= agent.amount_of_water_needed  # AQUIIIIII!
                                if model.verbose == True:
                                    print("Agent {} have overriden manager's decision withdrawing {} m³/year.". format(
                                        agent.unique_id, agent.amount_of_water_withdrawn))
                    else:
                        agent.agent_water_scarcity = True
                        agent.amount_of_water_withdrawn = 0
                agent.life_time += 1  # Increase 1 year in water right life time
            if (section != list(grouped_agents.keys())[-1]):
                next_section = current_list_of_sections[current_list_of_sections.index(
                    section)-len(current_list_of_sections)+1]
                model.available_water_per_section[str(
                    next_section)] += model.available_water_per_section[str(section)]

    def create_manager(self):
        m = ManagerAgent(len(self.G.nodes())+1, self)
        self.agents.add(m)

    def create_farmers_random_position(self):
        """
        Create farmers at random position in a linear graph.
        """
        i = 0
        while (i < self.number_of_farmers_to_create):
            # Returns a list with the random node
            random_node = random.sample(list(linear_graph.nodes()), 1)
            # Check whether cell is empty. If so, place agent
            if (len(self.grid.get_cell_list_contents(random_node)) == 0):
                f = FarmerAgent(self)
                self.agents.add(f)
                self.grid.place_agent(f, random_node[0])
                i += 1

    def allocate_all_water_to_section_one(self):
        """
        This function is used by withdraw_water and compose logit for water balance
        """
        init_water = self.init_water_available_per_section.copy()
        # sum total water available in m³/year
        total_water_available = sum(init_water.values())
        actual_water_available = {}
        for i in init_water:
            if (i == '1'):
                actual_water_available[i] = total_water_available
            else:
                actual_water_available[i] = 0
        return actual_water_available

    def create_human_supply(self):
        h = HumanSupplyAgent(self)
        self.agents.add(h)
        self.grid.place_agent(h, 10)  # place human supply on position 10

    def reset_water_available_for_current_year(self):
        self.available_water_per_section = self.allocate_all_water_to_section_one().copy()
        
    def collect_agents_data(self):
        # print(self.agents.select(lambda agent: agent.type == "farmer").get("atr_name))
        pass

    def step(self):
        """ Execute the step of all agents, one at a time. At the end advance model by one step """
        
        # Preparation
        if (self.time == 0):
            self.create_manager()
            """Manager define the water policy"""
            self.create_human_supply()
        self.create_farmers_random_position()

        # Run steps
        """Agents perform their steps and request their water rights"""
        self.agents.select(agent_type=FarmerAgent).do('step_stage_one')
        self.agents.select(agent_type=HumanSupplyAgent).do('step')
        self.agents.select(agent_type=ManagerAgent).do('step')
        self.withdraw_water()
        self.agents.select(agent_type=FarmerAgent).do('step_stage_two')
        self.collect_agents_data()

        # Advance time
        self.time += 1

    def run_model(self, step_count=3):
        for i in range(step_count):
            print("-------------- \n" +
                  "Initiating year n. " + str(i+1) + "\n" +
                  "--------------")
            self.step()


"Generate Linear Graph with NX"
linear_graph = DataPreparation.generate_edges_linear_graph(
    number_of_sections=15, number_of_nodes=20)

"Initial conditions"
# Values in m³/year
water_restriction_coef = 1
available_water_per_section = {
    '1': 764.5*4320*water_restriction_coef,
    '2': 808.1*4320*water_restriction_coef,
    '3': 752.4*4320*water_restriction_coef,
    '4': 825.1*4320*water_restriction_coef,
    '5': 784.2*4320*water_restriction_coef,
    '6': 680.0*4320*water_restriction_coef,
    '7': 646.7*4320*water_restriction_coef,
    '8': 569.9*4320*water_restriction_coef,
    '9': 518.3*4320*water_restriction_coef,
    '10': 435.9*4320*water_restriction_coef,
    '11': 377.8*4320*water_restriction_coef,
    '12': 344.2*4320*water_restriction_coef,
    '13': 265.9*4320*water_restriction_coef,
    '14': 261.8*4320*water_restriction_coef,
    '15': 305.0*4320*water_restriction_coef,
}

mnl_parameters = pd.read_excel('mnl_farmer_parameters.xlsx', index_col=0)
crop_parameters = pd.read_csv('crop_parameters.csv', index_col=0)

"Run model"
number_of_steps = 3
n_farmers_to_create_per_year = 2

model = WaterAllocationModel(
    linear_graph,  # linear graph created using nx
    available_water_per_section,  # dictionary with available water for 15 sections
    n_farmers_to_create_per_year,  # fixed number of farmers created per year
    mnl_parameters,  # mean and sd for normal and lognormal parameters distributions
    crop_parameters  # data including irrigation volume and revenue
)
model.run_model(step_count=number_of_steps)
