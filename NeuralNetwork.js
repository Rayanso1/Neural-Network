const real_values = [-1,1,0,0.5,-0.5]

const L0_inputs = [0.4, 0.3, 0.6, 0.1, 1]

const L1_biases = [0.4, 0.1, 0.2, 0.25, 0.5]

const L1_N1_weights = [0.2, 0.92, 0.5, 0.2, 0.3]
const L1_N2_weights = [0.3, 0.3, 0.3, 0.23, 0.2]
const L1_N3_weights = [0.2, 0.2, 1, 1, 0.4]
const L1_N4_weights = [0.95, 0.2, 0.2, 0.2, 0.6]
const L1_N5_weights = [0.9, 0.1, 0.9, 0.2, 0.1]

const L1_neurons = [L1_N1_weights,L1_N2_weights,L1_N3_weights,L1_N4_weights,L1_N5_weights]

const eta = 0.1
const e = 2.71828

function scaling_function(x) {

    return(eval((e**x - e**-x)/(e**x + e**-x)))

}

function scaling_function_derivative(x) {

    return(eval(1 - scaling_function(x)**2))

}

function single_node_calculator(array_neuron,bias_index,n=0,base_sum=0) {

    if (n < array_neuron.length) {

        let scaled_sum = (base_sum + (array_neuron[n]*L0_inputs[n]))

        return(single_node_calculator(array_neuron,bias_index,n+1,scaled_sum))
    }

    return(scaling_function(base_sum + L1_biases[bias_index]))

}


function neural_training(array_list_weights, biases, iterations, inputs=L0_inputs, real_results=real_values) {

    if (iterations > 0) {

        let results = []

        array_list_weights.forEach((array_neuron,index) => 
    
        {
        
        results.push(single_node_calculator(array_neuron,index))

        });

        console.log(results)

        let result_derivatives_biases = []

        let result_derivatives_weights_1 = []
        let result_derivatives_weights_2 = []
        let result_derivatives_weights_3 = []
        let result_derivatives_weights_4 = []
        let result_derivatives_weights_5 = []

        let weights_derivatives = [result_derivatives_weights_1, result_derivatives_weights_2, result_derivatives_weights_3, result_derivatives_weights_4, result_derivatives_weights_5]


        results.forEach((placeholder,index) => 
            
            {

            let node_index = index
            let intermidiate_derivatives = 0

            results.forEach((placeholder,index) => 
    
                {
        
                let c_index = index
                    

                intermidiate_derivatives += (eval(scaling_function_derivative((array_list_weights[node_index])[c_index] * inputs[c_index] + biases[node_index])*2*(results[c_index] - real_results[c_index])))
        
                });

            result_derivatives_biases.push(intermidiate_derivatives/(biases.length))

        }),
        
        results.forEach((placeholder,index) => 
            
            {
        
            let node_index = index
        
            results.forEach((placeholder,index) => 
            
                {
        
                let c_index = index

                weights_derivatives[node_index].push(eval(inputs[c_index]  *  scaling_function_derivative((array_list_weights[node_index])[c_index] * inputs[c_index] + biases[node_index] )  *  2*(results[node_index] - real_results[node_index])))
                
                });
            
            });

        
        
        let new_biases = []

        let N1_new_weights = []
        let N2_new_weights = []
        let N3_new_weights = []
        let N4_new_weights = []
        let N5_new_weights = []
            
        let new_weights = [N1_new_weights, N2_new_weights, N3_new_weights, N4_new_weights, N5_new_weights]
        
        weights_derivatives.forEach((placeholder,index) => {

            let list_index = index

            weights_derivatives[list_index].forEach((placeholder, index) => {

                let list_list_index = index

                new_weights[index].push((array_list_weights[list_index])[list_list_index] - (eta * weights_derivatives[list_index][list_list_index]))

            })

        }
        )

        result_derivatives_biases.forEach((placeholder,index) => {

            new_biases.push(biases[index] - (eta*(result_derivatives_biases[index])))
        }
        )

        return(neural_training(new_weights, new_biases, iterations-1))

    }

    return{
        weights: array_list_weights,
        biases: biases
    }

    

}

let trained_w_and_b = neural_training(L1_neurons, L1_biases, 5000)

let results = []

trained_w_and_b.weights.forEach((array_neuron,index) => 
    
    {
    
    results.push(single_node_calculator(array_neuron,index))

    });

console.log(results)
console.log(trained_w_and_b)