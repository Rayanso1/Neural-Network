"use strict";

function uniform_distribution() {

    return(Math.random()*(max-min) + min)

}

function synthetic_data(amount, size) {

    if (amount > 0) {
    
        let a = uniform_distribution()

        let x = [a]

        let inputs_array = x
        let outputs_array = [(a**2) + 1]

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

                    weights_layers[L_index - 1][i].push((uniform_distribution()) * Math.sqrt(6 / layers_lengths[L_index - 1]))
                }
            }
            return(xavier_initialization(L_index + 1))
        }
    }
    xavier_initialization(1)
}

function scaling_function(x) {

    return(Math.max(0,x))

}

function scaling_function_derivative(x) {

    if (x >= 0) {return 1}
    else {return(0)}
}

function single_node_calculator(weights, bias_index, biases, inputs, condition, n=0, base_sum=0,) {

    if (n < weights.length) {

        let intermetiate_sum = base_sum + (weights[n]*inputs[n])

        return(single_node_calculator(weights, bias_index, biases, inputs, condition, n+1, intermetiate_sum))
    }
    if (condition = -1) {
        unscaled_final_results.push(base_sum + biases[bias_index])
    }
    return(scaling_function(base_sum + biases[bias_index]))

}

const max = 1
const min = 0
const fs = require("fs")
const eta = 0.01
const input_size = 1
const output_size = 1
const layers_lengths = [input_size,5, output_size]
const nbr_iterations = 10_000_000
const data_amount = nbr_iterations + 1
const data_frequency = 1000
const batch_size = 10

var weights_layers = []
var biases_layers = []

w_and_b_arrays(layers_lengths)

console.log(weights_layers)

let synthetic_inputs = []
let synthetic_outputs = []

synthetic_data(data_amount, input_size)

let all_errors = []

var unscaled_final_results = []

var weights_batch =  []
var biases_batch = []

weights_layers.forEach((layer, index) => {

    let L_index = index
    weights_batch[L_index] = []

    layer.forEach((weights, index) => {

        let N_index = index
        weights_batch[L_index][N_index] = []

        weights.forEach((placeholder, index) => {

            let C_index = index
            weights_batch[L_index][N_index][C_index] = 0
        })
    })
})

biases_layers.forEach((layer, index) => {

    let L_index = index
    biases_batch[L_index] = []

    layer.forEach((placeholder, index) => {

        let N_index = index
        biases_batch[L_index][N_index] = 0
    }) 
})

function every_layer_neural_training(inputs, target_values, iterations) {

    if (iterations > 0) {

        if ((nbr_iterations - iterations + 1) % batch_size == 0) {

            weights_layers.forEach((layer, index) => {

                let L_index = index

                layer.forEach((weights, index) => {

                    let N_index = index

                    weights.forEach((placeholder, index) => {

                        let C_index = index
                        weights_layers[L_index][N_index][C_index] += -(eta * weights_batch[L_index][N_index][C_index])
                        weights_batch[L_index][N_index][C_index] = 0
                    })
                })
            })

            biases_layers.forEach((layer, index) => {

                let L_index = index

                layer.forEach((placeholder, index) => {

                    let N_index = index
                    biases_layers[L_index][N_index] += -(eta * biases_batch[L_index][N_index])
                    biases_batch[L_index][N_index] = 0
                }) 
            })
        }

        let results = [inputs]

        unscaled_final_results = []

        weights_layers.forEach((weights_arrays, index) => {

            let L_index = index
            let layer_results = []
            let previous_results = results[results.length - 1]

            weights_arrays.forEach((weights_values, index) => {

                layer_results.push(single_node_calculator(weights_values, index, biases_layers[L_index], previous_results, L_index - weights_layers.length))

            })
            results.push(layer_results)
        })

        let errors_sum = 0
        results[results.length - 1].forEach((result, index) => {errors_sum += Math.abs(result - target_values[index])})
        if ((iterations % data_frequency) == 0) {all_errors.push(errors_sum)}

        if (Number.isInteger(Math.log10((nbr_iterations - iterations))) ) {
            console.log("iterations: ",nbr_iterations-iterations,"\n")
            console.log("inputs: ",inputs)
            console.log("results: ",results[results.length - 1])
            console.log("target_values: ",target_values)
            console.log("error: ",errors_sum,"\n")
        }

        let cost_function = []

        results[results.length - 1].forEach((result, index) => {
            
            cost_function.push(2*(results[results.length - 1][index] - target_values[index]) * scaling_function_derivative(unscaled_final_results[index]))
        }
        )

        function layers_iterator(cost_function, L_index) {

            if (L_index >= 0) {
    
                weights_layers[L_index].forEach((placeholder, index) => {
    
                    let N_index = index
        
                    weights_layers[L_index][N_index].forEach((placeholder, index) => {
        
                        let C_index = index
                        weights_batch[L_index][N_index][C_index] += (cost_function[N_index] * results[L_index][C_index])
                    })
                })

                weights_layers[L_index].forEach((placeholder, index) => {

                    let N_index = index
                    biases_batch[L_index][N_index] += (cost_function[N_index])
                    
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
                    new_cost_function[C_index] = weights_x_erros_sum  * scaling_function_derivative(results[L_index][C_index])
                })
                
                layers_iterator(new_cost_function, L_index - 1)
            }
        }
        layers_iterator(cost_function, weights_layers.length - 1)

        return(every_layer_neural_training(synthetic_inputs[iterations - 1], synthetic_outputs[iterations - 1], iterations - 1))
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
every_layer_neural_training(synthetic_inputs[nbr_iterations], synthetic_outputs[nbr_iterations], nbr_iterations)
console.timeEnd()

console.log(results)

let JSON_array = JSON.stringify(all_errors, null, 2)
fs.writeFileSync("output.json",JSON_array,"utf8",)