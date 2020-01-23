

#include "Spike/Spike.hpp"
#include <stdlib.h>
#include <string>
#include <iostream>
#include <vector>
#include <fstream>
#include <sstream>
#include <getopt.h>

//#include <boost/filesystem.hpp>
#include<sys/stat.h>
using namespace std;

//this is required to print out EXCITATORY_NEURONS as a bug check
std::ostream& operator<<(std::ostream& os, const std::vector<int> &EXCITATORY_NEURONS){
	for (auto const& i: EXCITATORY_NEURONS) {
		os << i << " ";
	}
	return os;
}

int main (int argc, char *argv[]){



	string experimentName = string("image_four_layer_poisson_test");

	/*
	CHOOSE SIMULATION COMPONENTS
	*/

	//Create an instance of the model
	SpikingModel* model = new SpikingModel();

	//Set up the simulator with a timestep at which the neuron, synapse and STDP properties will be calculated
	float timestep = 0.0001; //this is calculated in seconds, therefore 0.0001 = 0.1ms
	model->SetTimestep(timestep);

	//Choose an input neuron type
	ImagePoissonInputSpikingNeurons* input_neurons = new ImagePoissonInputSpikingNeurons();
	//Choose a neuron type
	LIFSpikingNeurons* lif_spiking_neurons = new LIFSpikingNeurons();
	//Choose a synapse type
	ConductanceSpikingSynapses* conductance_spiking_synapses = new ConductanceSpikingSynapses();

	//Allocate these components to the simulator
	model->input_spiking_neurons = input_neurons;
	model->spiking_neurons = lif_spiking_neurons;
	model->spiking_synapses = conductance_spiking_synapses;

	/*
	ADD ACTIVITY MONITORS AND PLASTICITY RULES
	*/

	SpikingActivityMonitor* input_spike_monitor = new SpikingActivityMonitor(input_neurons);
	SpikingActivityMonitor* spike_monitor = new SpikingActivityMonitor(lif_spiking_neurons);
	//here in previous model there were also advanced parameter options

	//Allocate them to the simulator
	model->AddActivityMonitor(spike_monitor);
	model->AddActivityMonitor(input_spike_monitor);

	//STDP parameters
	evans_stdp_plasticity_parameters_struct STDP_PARAMS;
	// Below are the decay rates of the variables for learning: Pre/Post synaptic activities C and D (See Ben Evans)
	float decay_term_tau_C = 0.05; //aki_paper = 0.005 // 0.3(In Ben's model, tau_C/tau_D = 0.003/0.005 v 0.015/0.025 v 0.075/0.125, and the first one produces the best result)
	float decay_term_tau_D = 0.05; //aki_paper = 0.005
	float learning_rate_rho = 0.1f; //0.1 is default
	float synaptic_neurotransmitter_concentration_alpha_C = 0.5; //0.5 is default
	//(Perrinet et al, 2001) On one side, we’ll consider a pool of emitters (corresponding to the docked vesicles
	//containing neuro transmitter) quantified by their relative concentration C. 
	//This quantity is triggered by presynaptic spikes but the pool is limited (leading to synaptic depression). 
	float model_parameter_alpha_D = 0.5;
	//(Perrinet et al, 2001) On the other side, we’ll consider a pool of receivers (corresponding to the sensitivity of postsynaptic sites and that appear to be calcium related mechanism) similarly quantified by D. 
	//This quantity is mediated by postsynaptic spikes and may be related to synaptic facilitation. Referred to in Ben Evan's paper as proportion of unblocked NMDA receptors. 

	//Allocate them to the simulator
	STDP_PARAMS.decay_term_tau_C = decay_term_tau_C;
	STDP_PARAMS.decay_term_tau_D = decay_term_tau_D;
	STDP_PARAMS.model_parameter_alpha_D = model_parameter_alpha_D;
	STDP_PARAMS.synaptic_neurotransmitter_concentration_alpha_C = synaptic_neurotransmitter_concentration_alpha_C;
	STDP_PARAMS.learning_rate_rho = learning_rate_rho;
	EvansSTDPPlasticity* evans_stdp = new EvansSTDPPlasticity(conductance_spiking_synapses, lif_spiking_neurons, input_neurons, &STDP_PARAMS);
	model->AddPlasticityRule(evans_stdp);

	/* 
	SETUP PROPERTIES AND CREATE NETWORK
	Note: 
    All Neuron, Synapse and STDP types have associated parameters structures.
    These structures are defined in the header file for that class and allow us to set properties.
    */


	// SETTING UP INPUT NEURONS

	cout<<"Adding input neurons \n"<<endl;

	//loading the correct files
	string filepath = "../Data/MatlabGaborFilter/Inputs_Gisi_BO/";
	string filelist = "FileList.txt"; 
	string inputs_for_test_name = "Inputs_Gisi_BO";
	float max_FR_of_input_Gabor = 100.0f; // Hz
	//try this line above set_up_rates ***
	// Creating an input neuron parameter structure
	image_poisson_input_spiking_neuron_parameters_struct * image_poisson_input_spiking_group_params = new image_poisson_input_spiking_neuron_parameters_struct();
	//input_neurons->set_up_rates(filelist.c_str(), "FilterParameters.txt", (filepath+inputs_for_test_name+"/").c_str(), max_FR_of_input_Gabor);
	input_neurons->set_up_rates(filelist.c_str(), "FilterParameters.txt", filepath.c_str(), max_FR_of_input_Gabor);
	image_poisson_input_spiking_group_params->rate = 30.0f;
	input_neurons->AddGroupForEachGaborType(image_poisson_input_spiking_group_params);
	//currently not copied over equalize_rates fucntion, do not use ***
	//float equalize_target = 0.19f;
	//equalize_rates(input_neurons, equalize_target);
	
	cout<<"Input Neurons added \n"<<endl;
	
    cout<<"creating LIF neurons \n"<<endl;



    // Creating an LIF parameter structure for an excitatory neuron population and an inhibitory
    lif_spiking_neuron_parameters_struct * excitatory_population_params = new lif_spiking_neuron_parameters_struct();
    excitatory_population_params->group_shape[0] = 64;
	excitatory_population_params->group_shape[1] = 64;
    excitatory_population_params->resting_potential_v0 = -0.074f;
    excitatory_population_params->threshold_for_action_potential_spike = -0.053f;
	excitatory_population_params->somatic_capacitance_Cm = 500.0*pow(10, -12);
	excitatory_population_params->somatic_leakage_conductance_g0 = 25.0*pow(10, -9);

	lif_spiking_neuron_parameters_struct * inhibitory_population_params = new lif_spiking_neuron_parameters_struct();
    inhibitory_population_params->group_shape[0] = 32;
    inhibitory_population_params->group_shape[1] = 32;
    inhibitory_population_params->resting_potential_v0 = -0.082f;
    inhibitory_population_params->threshold_for_action_potential_spike = -0.053f;
    inhibitory_population_params->somatic_capacitance_Cm = 214.0*pow(10, -12);
    inhibitory_population_params->somatic_leakage_conductance_g0 = 18.0*pow(10, -9);


	//Set up the number of layers here
	const int number_of_layers = 4;
	//create empty inhib and exhit vectors
	vector<int> EXCITATORY_NEURONS;
	vector<int> INHIBITORY_NEURONS;


	// SETTING UP NEURON GROUP

	for (int l=0; l<number_of_layers; l++){
		EXCITATORY_NEURONS.push_back(model->AddNeuronGroup(excitatory_population_params));
		INHIBITORY_NEURONS.push_back(model->AddNeuronGroup(inhibitory_population_params));
		cout<<"Neuron Group "<<EXCITATORY_NEURONS[l]<<": Excitatory layer "<<l<<endl;
		cout<<"Neuron Group "<<INHIBITORY_NEURONS[l]<<": Inhibitory layer "<<l<<endl;
	}

	cout<<"LIF neurons added \n"<<endl;

	cout<<"Creating synapses \n"<<endl;


  	// SETTING UP SYNAPSES
    // Creating a synapses parameter structure for connections from the input neurons to the excitatory neurons
    conductance_spiking_synapse_parameters_struct* input_to_excitatory_parameters = new conductance_spiking_synapse_parameters_struct();
    input_to_excitatory_parameters->weight_range[0] = 0.0f;   // Create uniform distributions of weights [0.5, 10.0]
    input_to_excitatory_parameters->weight_range[1] = 1.0f;
    input_to_excitatory_parameters->weight_scaling_constant = excitatory_population_params->somatic_leakage_conductance_g0;
    input_to_excitatory_parameters->delay_range[0] = timestep;    // Create uniform distributions of delays [1 timestep, 5 timesteps]
    input_to_excitatory_parameters->delay_range[1] = timestep;
    input_to_excitatory_parameters->reversal_potential_Vhat = 0.0; //Volts
    input_to_excitatory_parameters->gaussian_synapses_per_postsynaptic_neuron = 90;
    input_to_excitatory_parameters->gaussian_synapses_standard_deviation = 10.0;
    input_to_excitatory_parameters->decay_term_tau_g = 0.15;
    input_to_excitatory_parameters->connectivity_type = CONNECTIVITY_TYPE_GAUSSIAN_SAMPLE;

    

    // Creating a set of synapse parameters for connections from the excitatory neurons to the inhibitory neurons
    conductance_spiking_synapse_parameters_struct * excitatory_to_inhibitory_parameters = new conductance_spiking_synapse_parameters_struct();
    excitatory_to_inhibitory_parameters->weight_range[0] = 0.0f;
    excitatory_to_inhibitory_parameters->weight_range[1] = 1.0f;
    excitatory_to_inhibitory_parameters->weight_scaling_constant = inhibitory_population_params->somatic_leakage_conductance_g0;
    excitatory_to_inhibitory_parameters->delay_range[0] = 5.0*timestep;
    excitatory_to_inhibitory_parameters->delay_range[1] = 0.1;
	excitatory_to_inhibitory_parameters->reversal_potential_Vhat = 0.0;
    excitatory_to_inhibitory_parameters->gaussian_synapses_per_postsynaptic_neuron = 60;
    excitatory_to_inhibitory_parameters->gaussian_synapses_standard_deviation = 8.0;
    excitatory_to_inhibitory_parameters->decay_term_tau_g = 0.005;
    excitatory_to_inhibitory_parameters->connectivity_type = CONNECTIVITY_TYPE_GAUSSIAN_SAMPLE;


    //Creating a set of synapse parameters for lateral connections between excitatory and excitatory neurons in the same layer
    conductance_spiking_synapse_parameters_struct * excitatory_to_excitatory_parameters = new conductance_spiking_synapse_parameters_struct();
    excitatory_to_excitatory_parameters->weight_range[0] = 0.0f;
    excitatory_to_excitatory_parameters->weight_range[1] = 1.0f;
    excitatory_to_excitatory_parameters->weight_scaling_constant = excitatory_population_params->somatic_leakage_conductance_g0;
	excitatory_to_excitatory_parameters->delay_range[0] = 5.0 * timestep;
	excitatory_to_excitatory_parameters->delay_range[1] = 0.1;
	//excitatory_to_excitatory_parameters->max_number_of_connections_per_pair = 1;
	excitatory_to_excitatory_parameters->gaussian_synapses_per_postsynaptic_neuron = 60;
	excitatory_to_excitatory_parameters->plasticity_vec.push_back(evans_stdp);
	excitatory_to_excitatory_parameters->reversal_potential_Vhat = 0.0;
	excitatory_to_excitatory_parameters->decay_term_tau_g = 0.15;
	excitatory_to_excitatory_parameters->connectivity_type = CONNECTIVITY_TYPE_GAUSSIAN_SAMPLE;


    // Creating a set of synapse parameters from the inhibitory neurons to the excitatory neurons
    conductance_spiking_synapse_parameters_struct * inhibitory_to_excitatory_parameters = new conductance_spiking_synapse_parameters_struct();
    inhibitory_to_excitatory_parameters->weight_range[0] = 0.0f;//-5.0f;
    inhibitory_to_excitatory_parameters->weight_range[1] = 1.0f;//-2.5f;
    inhibitory_to_excitatory_parameters->weight_scaling_constant = excitatory_population_params->somatic_leakage_conductance_g0;
    inhibitory_to_excitatory_parameters->delay_range[0] = 5.0*timestep;
    inhibitory_to_excitatory_parameters->delay_range[1] = 0.1;//3.0f*pow(10, -3);
    inhibitory_to_excitatory_parameters->gaussian_synapses_per_postsynaptic_neuron = 60;
    inhibitory_to_excitatory_parameters->gaussian_synapses_standard_deviation = 4.0;
    inhibitory_to_excitatory_parameters->reversal_potential_Vhat = -70.0*pow(10, -3);
    inhibitory_to_excitatory_parameters->decay_term_tau_g = 0.005;
    inhibitory_to_excitatory_parameters->connectivity_type = CONNECTIVITY_TYPE_GAUSSIAN_SAMPLE;

    int max_number_of_connections_per_pair = 2; // The maximum number of connections refers to multiple synaptic contacts pre->post

	// CREATING SYNAPSES
    // Note: Input neuron populations cannot be post-synaptic on any synapse
    // When creating synapses, the ids of the presynaptic and postsynaptic populations are all that are required

    float gaussian_synapses_standard_deviation_E2E_FF[number_of_layers-1] = {20.0, 20.0, 20.0}; // List for each layer, can be customized
    float layerwise_biological_conductance_scaling_constant_lambda_E2E_FF[number_of_layers-1] = {
		0.625f,  //0.625f gisi value
		0.5f, //0.5f gisi value
		0.75f};  //0.75f gisi value	



    
    for (int l=0; l<number_of_layers; l++){

		//input layer
		if(l==0){
			cout<<"input layer\n"<<endl;
		  	model->AddSynapseGroupsForNeuronGroupAndEachInputGroup(
		  		EXCITATORY_NEURONS[l], input_to_excitatory_parameters);
		}
		else{

			cout<<"layer "<<l<<"\n"<<endl;
			excitatory_to_excitatory_parameters->gaussian_synapses_standard_deviation = gaussian_synapses_standard_deviation_E2E_FF[l-1];
			excitatory_to_excitatory_parameters->weight_scaling_constant = layerwise_biological_conductance_scaling_constant_lambda_E2E_FF[l-1];
			for (int connection_number = 0; connection_number < max_number_of_connections_per_pair; connection_number++){
				model->AddSynapseGroup(EXCITATORY_NEURONS[l-1], EXCITATORY_NEURONS[l], excitatory_to_excitatory_parameters);
			}
			cout<<"here we would add e to e \n"<<endl;
			//bug check the contents of EXCITATORY_NEURONS
			cout<<"check EXCITATORY_NEURONS content: \n"<<endl;
			std::cout << EXCITATORY_NEURONS[l]<<"\n";
			
		}

		cout<<"add e to i synapses"<<endl;
		//excitatory_to_excitatory_parameters.gaussian_synapses_standard_deviation = gaussian_synapses_standard_deviation_E2E_FF[l-1];
		//excitatory_to_excitatory_parameters.weight_scaling_constant = layerwise_biological_conductance_scaling_constant_lambda_E2E_FF[l-1];
		//for (int connection_number = 0; connection_number < max_number_of_connections_per_pair; connection_number++){
			//model->AddSynapseGroup(EXCITATORY_NEURONS[l-1], EXCITATORY_NEURONS[l], excitatory_to_excitatory_parameters);
		model->AddSynapseGroup(EXCITATORY_NEURONS[l], INHIBITORY_NEURONS[l], excitatory_to_inhibitory_parameters);
		model->AddSynapseGroup(INHIBITORY_NEURONS[l], EXCITATORY_NEURONS[l], inhibitory_to_excitatory_parameters);
	}

	cout<<"Synapses created \n";

    /*
      RUN THE SIMULATION
    */

    //const float presentation_time_per_stimulus_per_epoch_test = 2.0f; // seconds


    model->finalise_model();

    string exppath = "/Users/hakunahahannah/Documents/Projects/Spike/Build/output/" + experimentName;
	string initpath = "/Users/hakunahahannah/Documents/Projects/Spike/Build/output/" + experimentName + "/initial";
	string trainpath = "/Users/hakunahahannah/Documents/Projects/Spike/Build/output/" + experimentName + "/training";
	string testpath = "/Users/hakunahahannah/Documents/Projects/Spike/Build/output/" + experimentName + "/testing";

	mkdir(exppath.c_str(), ACCESSPERMS);
	mkdir(initpath.c_str(), ACCESSPERMS);
	mkdir(trainpath.c_str(), ACCESSPERMS);
	mkdir(testpath.c_str(), ACCESSPERMS);

	//what does this line do? function hidden in backend ***
	input_neurons->copy_rates_to_device();

    for (int stimulus_index = 0; stimulus_index < input_neurons->total_number_of_input_stimuli; stimulus_index++){
    	// for (int stimulus_index = 0; stimulus_index < 2; stimulus_index++){
    	input_neurons->select_stimulus(stimulus_index);
    	//model->run(presentation_time_per_stimulus_per_epoch_test); // false for plasticity_on
    	model->run(10.0f);
    	spike_monitor->save_spikes_as_txt(trainpath + "/");
    	input_spike_monitor->save_spikes_as_txt(initpath + "/");
    	model->spiking_neurons->reset_state();
    	model->spiking_synapses->save_connectivity_as_txt(initpath + "/initial_");
    	model->spiking_synapses->reset_state();
    	input_spike_monitor->reset_state();
    	spike_monitor->reset_state();
	}
	model->monitors_vec.pop_back();


	//float simtime = 50.0f;
	//model->run(simtime);

    //spike_monitor->save_spikes_as_txt(trainpath + "/");
    //input_spike_monitor->save_spikes_as_txt(initpath + "/");
    //model->spiking_synapses->save_connectivity_as_txt(trainpath + "/");


	return 0;
}











