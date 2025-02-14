const real_values = [Math.random(),Math.random(),Math.random(),Math.random(),Math.random()]

const L0_inputs = [0.5, 0.5, -1, -1, -0.5]

const L1_biases = [0, 0, 0, 0, 0]

const L2_biases = [0, 0, 0, 0, 0]

const L1_weights = [

    [0.2, 0.92, 0.5, 0.2, 0.3],
    [0.3, 0.3, 0.3, 0.23, 0.2],
    [0.2, 0.2, 1, 1, 0.4],
    [0, 0, 0, 0, 0],
    [0.9, 0.1, 0.9, 0.2, 0.1],

]

const L2_weights = [

    [0.2, 0.92, 0.5, 0.2, 0.3],
    [0.3, 0.3, 0.3, 0.23, 0.2],
    [0.2, 0.2, 1, 1, 0.4],
    [0, 0, 0, 0, 0],
    [0.9, 0.1, 0.9, 0.2, 0.1]

]

const eta = 0.1
const e = 2.718281828459045235360287471352

function scaling_function(x) {

    return(eval((e**x - e**-x)/(e**x + e**-x)))

}

function scaling_function_derivative(x) {

    return(eval(1 - scaling_function(x)**2))

}

function single_node_calculator(array_neuron, bias_index, biases=L1_biases, inputs=L0_inputs, n=0, base_sum=0) {

    if (n < array_neuron.length) {

        let intermetiate_sum = (base_sum + (array_neuron[n]*inputs[n]))

        return(single_node_calculator(array_neuron, bias_index, biases, inputs, n+1, intermetiate_sum))
    }

    return(scaling_function(base_sum + biases[bias_index]))

}

let second_layer_inputs = []
L1_weights.forEach((array_neuron, index) => (second_layer_inputs.push(single_node_calculator(array_neuron, index, L1_biases, L0_inputs))))

const L1_inputs = second_layer_inputs

function neural_training(array_list_weights, biases, inputs, iterations, real_results=real_values) {

    if (iterations > 0) {

        let results = []

        array_list_weights.forEach((array_neuron, index) => 
    
            {
        
            results.push(single_node_calculator(array_neuron, index, biases, inputs))

            });

        console.log(results)

        let result_derivatives_biases = []
        let weights_derivatives = [[], [], [], [], []]

        results.forEach((placeholder,index) => 
            
            {

            let node_index = index
            let intermidiate_derivative = 0

            results.forEach((placeholder,index) => 
    
                {
        
                let c_index = index
                    

                intermidiate_derivative += (eval(scaling_function_derivative((array_list_weights[node_index])[c_index] * inputs[c_index] + biases[node_index])*2*(results[c_index] - real_results[c_index])))
        
                });

            result_derivatives_biases.push(intermidiate_derivative/(biases.length))

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
        let new_weights = [[], [], [], [], []]
                
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

        return(neural_training(new_weights, new_biases, inputs, iterations-1))

    }

    return{
        weights: array_list_weights,
        biases: biases
    }

    

}

console.time()
let trained_w_and_b = neural_training(L2_weights, L2_biases, L0_inputs, 5000)
console.timeEnd()

let final_results = []

trained_w_and_b.weights.forEach((array_neuron, index) => 
    
    {
    
    final_results.push(Number(single_node_calculator(array_neuron, index, trained_w_and_b.biases).toFixed(3)))

    });

console.log("rounded final resutls: ")
console.log(final_results)
console.log(trained_w_and_b)

let L1_inputs_derivative = []

let first_results = []

L1_weights.forEach((array_neuron, index) => 
    
    {
    
    first_results.push(single_node_calculator(array_neuron, index, L1_biases))

    });

trained_w_and_b.weights.forEach((placeholder, index) => {

    let intermediate_derivative = 0
    let node_index = index

    trained_w_and_b.weights[node_index].forEach((placeholder, index) => {
    
        let c_index = index
        intermediate_derivative += (eval(trained_w_and_b.weights[node_index][c_index] * scaling_function_derivative(trained_w_and_b.weights[node_index][c_index] * L0_inputs[c_index] + L1_biases[node_index]) *2*(first_results[node_index] - real_values[node_index])))

    })

    L1_inputs_derivative.push(intermediate_derivative/trained_w_and_b.weights[node_index].length)

})

console.log("L1_inputs_derivatives: ")
console.log(L1_inputs_derivative)