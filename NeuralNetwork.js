function synthetic_data(amount, size) {

    if (amount > 0) {

        let inputs_array = []
        let outputs_array = []
    
        for(let i = size; i > 0; i += -1) {
    
            let nbr = (2*Math.random() - 1)
            inputs_array.push(nbr)
            outputs_array.push(nbr/2)
        }
        synthetic_inputs.push(inputs_array)
        synthetic_outputs.push(outputs_array)

        return(synthetic_data(amount - 1, size))
    }
}

function w_and_b_arrays(layers_lengths) {

    function xavier_initialization(L_index) {

        if (layers_lengths.length > L_index) {

            weights_layers.push([])
            biases_layers.push([])

            for(let i = 0; i < layers_lengths[L_index]; i++) {

                weights_layers[L_index - 1].push([])
                biases_layers[L_index - 1].push(0)

                for(let j = 0; j < layers_lengths[L_index - 1]; j++) {

                    weights_layers[L_index - 1][i].push((2*Math.random() - 1) * Math.sqrt(6 / layers_lengths[L_index - 1]))
                }
            }
            return(xavier_initialization(L_index + 1))
        }
    }
    xavier_initialization(1)
}

function scaling_function(x) {

    return((e**x - e**-x)/(e**x + e**-x))

}

function scaling_function_derivative(x) {

    return(1 - scaling_function(x)**2)

}

function single_node_calculator(weights, bias_index, biases, inputs, n=0, base_sum=0) {

    if (n < weights.length) {

        let intermetiate_sum = base_sum + (weights[n]*inputs[n])

        return(single_node_calculator(weights, bias_index, biases, inputs, n+1, intermetiate_sum))
    }

    return(scaling_function(base_sum + biases[bias_index]))

}

const eta = 0.01
const e = 2.718281828459045235360287471352
const input_size = 1
const output_size = 1
const layers_lengths = [input_size, 50, output_size]
const nbr_iterations = 1_000_000
const data_amount = nbr_iterations + 1

let weights_layers = []
let biases_layers = []

w_and_b_arrays(layers_lengths)

let synthetic_inputs = []
let synthetic_outputs = []

synthetic_data(data_amount, input_size)

let all_errors = []

function every_layer_neural_training(weights_layers, biases_layers, inputs, target_values, iterations) {

    if (iterations > 0) {

        let results = [inputs]

        weights_layers.map((weights_arrays, index) => {

            let L_index = index
            let layer_results = []
            let previous_results = results[results.length - 1]

            weights_arrays.map((weights_values, index) => {

                layer_results.push(single_node_calculator(weights_values, index, biases_layers[L_index], previous_results))

            })
            results.push(layer_results)
        })

        if (Number.isInteger(Math.log10((nbr_iterations - iterations))) ) {
            console.log("iterations: ",nbr_iterations-iterations,"\n")
            console.log("inputs: ",inputs)
            console.log("results: ",results[results.length - 1])
            console.log("target_values: ",target_values)
            let errors_sum = 0
            results[results.length - 1].forEach((result, index) => {errors_sum += Math.abs(result - target_values[index])})
            all_errors.push(errors_sum * 10+"%")
            console.log("error: ",errors_sum * 10+"%","\n")
        }

        let cost_function = []
        results[results.length - 1].forEach((result, index) => cost_function.push(2 * (results[results.length - 1][index] - target_values[index])))

        let new_weights_layers =  []
        let new_biases_layers = []

        weights_layers.forEach((layer, index) => {

            let L_index = index
            new_weights_layers[L_index] = []

            layer.forEach((weights, index) => {

                let N_index = index
                new_weights_layers[L_index][N_index] = []

                weights.forEach((placeholder, index) => {

                    let C_index = index
                    new_weights_layers[L_index][N_index][C_index] = []
                })
            })
        })

        biases_layers.forEach((layer, index) => {

            let L_index = index
            new_biases_layers[L_index] = []

            layer.forEach((placeholder, index) => {

                let N_index = index
                new_biases_layers[L_index][N_index] = []
            }) 
        })

        function layers_iterator(cost_function, L_index) {

            if (L_index >= 0) {
    
                weights_layers[L_index].forEach((placeholder, index) => {
    
                    let N_index = index
        
                    weights_layers[L_index][N_index].forEach((placeholder, index) => {
        
                        let C_index = index
                        new_weights_layers[L_index][N_index][C_index] = weights_layers[L_index][N_index][C_index] - (eta * cost_function[N_index] * results[L_index][C_index])
                    })
                })

                weights_layers[L_index].forEach((placeholder, index) => {

                    let N_index = index
                    new_biases_layers[L_index][N_index] = biases_layers[L_index][N_index] - (eta * cost_function[N_index])

                })

                let new_cost_function = []

                weights_layers[L_index][0].forEach((placeholder, index) => {

                    let weights_x_erros_sum = 0
                    let C_index = index
                    new_cost_function.push([])

                    weights_layers[L_index].forEach((placeholder, index) => {
                    
                        let N_index = index 
                        weights_x_erros_sum += (weights_layers[L_index][N_index][C_index] * cost_function[N_index])
                    })
                    new_cost_function[C_index] = weights_x_erros_sum * scaling_function_derivative(results[L_index][C_index])
                })
                
                layers_iterator(new_cost_function, L_index - 1)
            }
        }
        layers_iterator(cost_function, weights_layers.length - 1)
        return(every_layer_neural_training(new_weights_layers, new_biases_layers, synthetic_inputs[iterations - 1], synthetic_outputs[iterations - 1], iterations - 1))
    }   

    else {
        results.weights = weights_layers
        results.biases = biases_layers
    }
}

let results = {
    weights: null,
    biases: null
}

console.time()
every_layer_neural_training(weights_layers, biases_layers, synthetic_inputs[nbr_iterations], synthetic_outputs[nbr_iterations], nbr_iterations)
console.timeEnd()

console.log(results)
console.log(all_errors)