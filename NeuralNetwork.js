"use strict";

function uniform_distribution(min, max) {

    return(Math.random()*(max-min) + min)

}

function normal_distribution(mean, stddev) {

    let z0 = Math.sqrt(-2 * Math.log(Math.random())) * Math.cos(2 * Math.PI * Math.random())

    return(z0 * stddev + mean)

}

function synthetic_data(amount) {

    if (amount > 0) {
    
        let a = uniform_distribution(min, max)

        let x = [a]

        let inputs_array = x
        let outputs_array = [function_to_approximate(x)]

        training_inputs.push(inputs_array)
        training_outputs.push(outputs_array)

        return(synthetic_data(amount - 1))
    }
}

function w_and_b_arrays(layers_lengths) {

    function initialization(L_index) {

        if (layers_lengths.length > L_index) {

            weights_layers.push([])
            biases_layers.push([])

            for(let i = 0; i < layers_lengths[L_index]; i++) {

                weights_layers[L_index - 1].push([])
                biases_layers[L_index - 1].push(0)

                for(let j = 0; j < layers_lengths[L_index - 1]; j++) {

                    if (activation_functions[L_index - 1] == ReLu) {weights_layers[L_index - 1][i][j] = (normal_distribution(0,Math.sqrt(2/(layers_lengths[L_index - 1]))))}
                    if (activation_functions[L_index - 1] == tanh) {weights_layers[L_index - 1][i][j] = (normal_distribution(0,Math.sqrt(2/(layers_lengths[L_index - 1] + layers_lengths[L_index]))))}
                    if (activation_functions[L_index - 1] == sigmoid) {weights_layers[L_index - 1][i][j] = (normal_distribution(0,2/(layers_lengths[L_index - 1])))}
                }
            }
            return(initialization(L_index + 1))
        }
    }
    initialization(1)
}


function single_node_calculator(L_index, weights, bias_index, biases, inputs, condition, n=0, base_sum=0,) {

    if (n < weights.length) {

        let intermetiate_sum = base_sum + (weights[n]*inputs[n])

        return(single_node_calculator(L_index, weights, bias_index, biases, inputs, condition, n+1, intermetiate_sum))
    }
    if (condition = -1) {
        unscaled_final_results.push(base_sum + biases[bias_index])
    }
    return(activation_functions[L_index](base_sum + biases[bias_index]))

}

const max = 1
const min = 0
const fs = require("fs")
const learning_rate = 0.001
const input_size = 28*28
const output_size = 10
const layers_lengths = [input_size, 256, 256, output_size]

const activation_functions = [ReLu, ReLu, sigmoid]
const activation_functions_derivatives = [ReLu_dx, ReLu_dx, sigmoid_dx]

const nbr_iterations = 100_000
const data_amount = 60_000
const data_frequency = 1
const batch_size = 20
const beta_1 = 0.9
const beta_2 = 0.999
const eta = 10**-8

const path = "" //path to the MNIST dataset

function loss_function(x,y) {} 
function loss_function_derivative(x,y) {return(2*(x-y))} //x: computed value, y: target value

function ReLu(x) {return(Math.max(0,x))}
function sigmoid(x) {return(1/(1+(Math.E**(-x))))}
function ReLu_dx(x) {if (x<0) {return(0)} else {return(1)}}
function sigmoid_dx(x) {return(sigmoid(x) * (1 - sigmoid(x)))}
function tanh(x) {return(Math.tanh(x))}
function tanh_dx(x) {return(1/(Math.cosh(x)**2))}


const use_synthetic_data = 0 //0 to use MNIST dataset //1 to use datasets from functions

function function_to_approximate(x) {return(Math.sin(5*x))}

var weights_layers = []
var biases_layers = []

w_and_b_arrays(layers_lengths)

if (use_synthetic_data == 1) {

    var training_inputs = []
    var training_outputs = []

    synthetic_data(data_amount)

}

else {

    var data_inputs = []
    var target_values = []

    var text = fs.readFileSync(path,"utf-8")
    var lines = text.split("\n")

    lines.shift()

    lines.forEach((line, index) => {

        let array = []

        for(let i = 0; i < 10; i++) {

            if (line.charAt(0) == i) {

                array.push(1)

            }

            else {

                array.push(0)

            }
        }

        target_values.push(array)

        let input_array = (line.split(",").map(Number))
        input_array.shift();

        let scaled_input_array = []

        input_array.forEach((number) => {scaled_input_array.push(number/255)})

        data_inputs.push(scaled_input_array)

    })

    console.log("File has been read.")

    var training_inputs = data_inputs
    var training_outputs = target_values

}

let all_errors = []
let all_results_targets = []

var unscaled_final_results = []

var weights_derivatives_batch =  []
var biases_derivatives_batch = []

var weights_derivatives_momentum = []
var biases_derivatives_momentum = []

var EWA_weights = []
var EWA_biases = []

var bias_correction_momentum_weights = []
var bias_correction_momentum_biases = []

var bias_correction_EWA_weights = []
var bias_correction_EWA_biases = []

weights_layers.forEach((layer, index) => {

    let L_index = index
    weights_derivatives_batch[L_index] = []
    weights_derivatives_momentum[L_index] = []
    EWA_weights[L_index] = []
    bias_correction_momentum_weights[L_index] = []
    bias_correction_EWA_weights[L_index] = []

    layer.forEach((weights, index) => {

        let N_index = index
        weights_derivatives_batch[L_index][N_index] = []
        weights_derivatives_momentum[L_index][N_index] = []
        EWA_weights[L_index][N_index] = []
        bias_correction_momentum_weights[L_index][N_index] = []
        bias_correction_EWA_weights[L_index][N_index] = []

        weights.forEach((placeholder, index) => {

            let C_index = index
            weights_derivatives_batch[L_index][N_index][C_index] = 0
            weights_derivatives_momentum[L_index][N_index][C_index] = 0
            EWA_weights[L_index][N_index][C_index] = 0
            bias_correction_momentum_weights[L_index][N_index][C_index] = 0
            bias_correction_EWA_weights[L_index][N_index][C_index] = 0
        })
    })
})

biases_layers.forEach((layer, index) => {

    let L_index = index
    biases_derivatives_batch[L_index] = []
    biases_derivatives_momentum[L_index] = []
    EWA_biases[L_index] = []
    bias_correction_momentum_biases[L_index] = []
    bias_correction_EWA_biases[L_index] = []

    layer.forEach((placeholder, index) => {

        let N_index = index
        biases_derivatives_batch[L_index][N_index] = 0
        biases_derivatives_momentum[L_index][N_index] = 0
        EWA_biases[L_index][N_index] = 0
        bias_correction_momentum_biases[L_index][N_index] = 0
        bias_correction_EWA_biases[L_index][N_index] = 0
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

                        weights_derivatives_momentum[L_index][N_index][C_index] = (beta_1 * weights_derivatives_momentum[L_index][N_index][C_index] + (1 - beta_1)*(weights_derivatives_batch[L_index][N_index][C_index]/batch_size))
                        EWA_weights[L_index][N_index][C_index] = (beta_2 * EWA_weights[L_index][N_index][C_index] + (1 - beta_2)*((weights_derivatives_batch[L_index][N_index][C_index]/batch_size)**2))
                        bias_correction_momentum_weights[L_index][N_index][C_index] = weights_derivatives_momentum[L_index][N_index][C_index]/(1-beta_1)
                        bias_correction_EWA_weights[L_index][N_index][C_index] = EWA_weights[L_index][N_index][C_index]/(1-beta_2)

                        weights_layers[L_index][N_index][C_index] += -(learning_rate * bias_correction_momentum_weights[L_index][N_index][C_index]/Math.sqrt(bias_correction_EWA_weights[L_index][N_index][C_index] + eta))
                        weights_derivatives_batch[L_index][N_index][C_index] = 0
                    })
                })
            })

            biases_layers.forEach((layer, index) => {

                let L_index = index

                layer.forEach((placeholder, index) => {

                    let N_index = index

                    biases_derivatives_momentum[L_index][N_index] = (beta_1 * biases_derivatives_momentum[L_index][N_index] + (1 - beta_1)*(biases_derivatives_batch[L_index][N_index]/batch_size))
                    EWA_biases[L_index][N_index] = (beta_2 * EWA_biases[L_index][N_index] + (1 - beta_2)*((biases_derivatives_batch[L_index][N_index]/batch_size)**2))
                    bias_correction_momentum_biases[L_index][N_index] = biases_derivatives_momentum[L_index][N_index]/(1-beta_1)
                    bias_correction_EWA_biases[L_index][N_index] = EWA_biases[L_index][N_index]/(1-beta_2)

                    biases_layers[L_index][N_index] += -(learning_rate * bias_correction_momentum_biases[L_index][N_index]/Math.sqrt(bias_correction_EWA_biases[L_index][N_index] + eta))
                    biases_derivatives_batch[L_index][N_index] = 0
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
                
                layer_results.push(single_node_calculator(L_index, weights_values, index, biases_layers[L_index], previous_results, L_index - weights_layers.length))

            })
            results.push(layer_results)
        })

        let errors_sum = 0
        results[results.length - 1].forEach((result, index) => {errors_sum += Math.abs(result - target_values[index])})
        if ((iterations % data_frequency) == 0) {all_errors.push(errors_sum)}

        if ((iterations % data_frequency) == 0) {all_results_targets.push(results[results.length - 1],target_values)}

        if (Number.isInteger(Math.log10((nbr_iterations - iterations))) ) {
            console.log("iterations: ",nbr_iterations-iterations,"\n")
            console.log("inputs: ",inputs)
            console.log("results: ",results[results.length - 1])
            console.log("target_values: ",target_values)
            console.log("error: ",errors_sum,"\n")
        }

        let cost_function = []

        results[results.length - 1].forEach((result, index) => {
                
            cost_function.push(loss_function_derivative(results[results.length - 1][index], target_values[index]) * activation_functions_derivatives[activation_functions_derivatives.length - 1](unscaled_final_results[index]))

        }
        )

        function layers_iterator(cost_function, L_index) {

            if (L_index >= 0) {
        
                weights_layers[L_index].forEach((placeholder, index) => {
        
                    let N_index = index
            
                     weights_layers[L_index][N_index].forEach((placeholder, index) => {
            
                        let C_index = index
                        weights_derivatives_batch[L_index][N_index][C_index] += (cost_function[N_index] * results[L_index][C_index])
                    })
                })

                weights_layers[L_index].forEach((placeholder, index) => {

                    let N_index = index
                    biases_derivatives_batch[L_index][N_index] += (cost_function[N_index])
                })

                if (L_index > 0) {

                    let new_cost_function = []
                    
                    weights_layers[L_index][0].forEach((placeholder, index) => {

                    let weights_x_erros_sum = 0
                    let C_index = index
                    new_cost_function.push([])

                    weights_layers[L_index].forEach((placeholder, index) => {
                        
                        let N_index = index 
                        weights_x_erros_sum += (weights_layers[L_index][N_index][C_index] * cost_function[N_index])
                    })

                    new_cost_function[C_index] = weights_x_erros_sum  * activation_functions_derivatives[L_index - 1](results[L_index][C_index])
                })

                layers_iterator(new_cost_function, L_index - 1)
                }
                else {}
            }
        }
        layers_iterator(cost_function, weights_layers.length - 1)

        return(every_layer_neural_training(training_inputs[(iterations - 1) % data_amount], training_outputs[(iterations - 1) % data_amount], iterations - 1))
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
every_layer_neural_training(training_inputs[nbr_iterations % data_amount], training_outputs[nbr_iterations % data_amount], nbr_iterations)
console.timeEnd()

let JSON_array_errors = JSON.stringify(all_errors, null, 2)
let JSON_array_parameters = JSON.stringify(results, null, 1)
let JSON_array_results = JSON.stringify(all_results_targets,null,2)

fs.writeFileSync("errors.json",JSON_array_errors,"utf8",)
fs.writeFileSync("parameters.json",JSON_array_parameters,"utf8",)
fs.writeFileSync("results_targets",JSON_array_results,"utf8",)

if (use_synthetic_data == 1) {

    let confirmation_data_input = []
    let confirmation_data_output = []

    const graphing_data_quantity = 1000

    function synthetic_data_2(amount) {

        if (amount > 0) {
    
            let a = min + (max-min)*(1/graphing_data_quantity)*(graphing_data_quantity-amount)

            let x = [a]

            let inputs_array = x
            let outputs_array = [function_to_approximate(x)]

            confirmation_data_input.push(inputs_array)
            confirmation_data_output.push(outputs_array)

            return(synthetic_data_2(amount - 1))
        }
    }

    synthetic_data_2(graphing_data_quantity)

    let computed_values = []

    for(let i = 0; i < graphing_data_quantity;  i++) {

        let result_by_layer = [confirmation_data_input[i]]

        weights_layers.forEach((weights_arrays, index) => {

        let L_index = index
        let layer_results = []
        let previous_results = result_by_layer[result_by_layer.length - 1]

        weights_arrays.forEach((weights_values, index) => {
        
            layer_results.push(single_node_calculator(L_index, weights_values, index, biases_layers[L_index], previous_results, L_index - weights_layers.length))

        })
        result_by_layer.push(layer_results)
        })
        computed_values.push(result_by_layer[result_by_layer.length-1])
    }

    let JSON_array_computed_y_values = JSON.stringify(computed_values, null, 2)

    let JSON_array_real_x_values = JSON.stringify(confirmation_data_input, null, 2)
    let JSON_array_real_y_values = JSON.stringify(confirmation_data_output, null, 2)

    fs.writeFileSync("computed_y_values.json",JSON_array_computed_y_values,"utf8",)
    fs.writeFileSync("real_x_values.json",JSON_array_real_x_values,"utf8",)
    fs.writeFileSync("real_y_values.json",JSON_array_real_y_values,"utf8",) 

}