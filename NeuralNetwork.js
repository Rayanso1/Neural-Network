const real_values = [2*Math.random()-1,2*Math.random()-1,2*Math.random()-1,2*Math.random()-1,2*Math.random()-1]

const L0_inputs = [-0.5, -0.5, 1, -1, 0.2]

const L1_biases = [0.2, 0.1, -0.1, 0.5, 0.1, -0.1, 0.5, 0.1]
const L2_biases = [0.49, 0.12, 0.4, -0.9, 0, -0.1, 0.5, 0.1]
const L3_biases = [0.44, 0.6, 0.2, 0.1, -0.1]

const L1_weights = [

    [-0.2, -0.92, -0.5, 0.2, 0.3],
    [0.3, -0.3, -0.3, -0.23, 0.3],
    [-0.2, -0.2, -1, 1, 0.3],
    [0.3, -0.1, 0.9, 0.11, 0.3],
    [-0.2, -0.2, -1, 1, 0.3],
    [0.3, -0.1, 0.9, 0.11, 0.3],
    [0.3, -0.3, -0.3, -0.23, 0.3],
    [-0.2, -0.92, -0.5, 0.2, 0.3],

]

const L2_weights = [

    [-0.2, 0.92, -0.5, 0.2, 0.3, -0.5, 0.2, 0.3],
    [0.3, 0.3, -0.3, -0.23, 0.2,-0.3, -0.23, 0.2],
    [0.2, -0.2, -1, 1, 0.1, -1, 1, 0.1],
    [0.5, 0.1, 0.2, -0.7, 0.1, 0.2, -0.7, 0.1],
    [0.5, 0.1, 0.2, -0.7, 0.2, 0.2, -0.7, 0.2],
    [0.3, 0.3, -0.3, -0.23, 0.2,-0.3, -0.23, 0.2],
    [0.2, -0.2, -1, 1, 0.1, -1, 1, 0.1],
    [0.5, 0.1, 0.2, -0.7, 0.1, 0.2, -0.7, 0.1],

]

const L3_weights = [

    [0.2, 0.92, 0.5, 0.2, -0.4, -0.2, -1, 1],
    [0.3, 0.3, 0.3, 0.23, 0.1, -0.2, -1, 1],
    [0.2, -0.2, 1, 1, 0.5, -0.2, -1, 1],
    [0.8, -0.1, 0.2, 0.3, 0.3, 0.92, -0.5, 0.2],
    [0.9, -0.1, 0.9, 0.2, 0.2, 0.92, -0.5, 0.2],

]

const weights_layers = [L1_weights, L2_weights, L3_weights]
const biases_layers = [L1_biases, L2_biases, L3_biases]

const eta = 0.1
const e = 2.718281828459045235360287471352

function scaling_function(x) {

    return(eval((e**x - e**-x)/(e**x + e**-x)))

}

function scaling_function_derivative(x) {

    return(eval(1 - scaling_function(x)**2))

}

function single_node_calculator(weights, bias_index, biases=L1_biases, inputs=L0_inputs, n=0, base_sum=0) {

    if (n < weights.length) {

        let intermetiate_sum = (base_sum + (weights[n]*inputs[n]))

        return(single_node_calculator(weights, bias_index, biases, inputs, n+1, intermetiate_sum))
    }

    return(scaling_function(base_sum + biases[bias_index]))

}

function every_layer_neural_training(weights_layers, biases_layers, inputs, target_values, iterations) {

    if (iterations > 0) {

        let results = []
        results.push(inputs)

        weights_layers.forEach((placeholder, index) => {

            let L_index = index
            let layer_results = []
            let intermetiate_results = results[results.length - 1]

            weights_layers[L_index].forEach((placeholder, index) => {

                let N_index = index

                layer_results.push(single_node_calculator(weights_layers[L_index][N_index], N_index, biases_layers[L_index], intermetiate_results))

            })
            results.push(layer_results)
        })
        console.log("results ",results[results.length - 1])

        let cost_function = []
        results[results.length - 1].forEach((result, index) => cost_function.push(2 * (results[results.length - 1][index] - target_values[index])))

        let weights_derivatives_layers = []
        let biases_derivatives_layers = []
        
        let new_weights_layers =  []
        let new_biases_layer = []

        weights_layers.forEach((layer, index) => {

            let L_index = index
            weights_derivatives_layers[L_index] = []
            new_weights_layers[L_index] = []

            layer.forEach((weights, index) => {

                let N_index = index
                weights_derivatives_layers[L_index][N_index] = []
                new_weights_layers[L_index][N_index] = []

                weights.forEach((placeholder, index) => {

                    let C_index = index
                    weights_derivatives_layers[L_index][N_index][C_index] = []
                    new_weights_layers[L_index][N_index][C_index] = []
                })
            })
        })

        biases_layers.forEach((layer, index) => {

            let L_index = index
            biases_derivatives_layers[L_index] = []
            new_biases_layer[L_index] = []

            layer.forEach((placeholder, index) => {

                let N_index = index
                biases_derivatives_layers[L_index][N_index] = []
                new_biases_layer[L_index][N_index] = []
            }) 
        })

        function layers_iterator(cost_function, L_index) {

            if (L_index >= 0) {
    
                weights_layers[L_index].forEach((placeholder, index) => {
    
                    let N_index = index
        
                    weights_layers[L_index][N_index].forEach((placeholder, index) => {
        
                        let C_index = index
                        weights_derivatives_layers[L_index][N_index][C_index] = (cost_function[N_index]*results[L_index][C_index])
                        //console.log(L_index," ",N_index," ",C_index)
                    })
                })

                weights_layers[L_index][0].forEach((placeholder, index) => {

                    let N_index = index
                    biases_derivatives_layers[L_index][N_index] = cost_function[N_index]

                })

                let new_cost_function = []

                weights_layers[L_index][0].forEach((placeholder, index) => {

                    let weights_x_erros_sum = 0
                    let C_index = index
                    new_cost_function.push([])

                    weights_layers[L_index].forEach((placeholder, index) => {
                    
                        let N_index = index 
                        weights_x_erros_sum += (weights_layers[L_index][N_index][C_index] * cost_function[N_index])
                        //console.log(L_index," ",N_index," ",C_index, "\n",)
                    })
                    new_cost_function[C_index] = (weights_x_erros_sum * scaling_function_derivative(results[L_index][C_index]))
                })
                
                layers_iterator(new_cost_function, L_index - 1)
            }
        }
        layers_iterator(cost_function, weights_layers.length - 1)

        weights_layers.forEach((placeholder, index) => {

            let L_index = index

            weights_layers[L_index].forEach((placeholder, index) => {

                let N_index = index

                weights_layers[L_index][N_index].forEach((placeholder, index) => {

                    let C_index = index
                    new_weights_layers[L_index][N_index][C_index] = (weights_layers[L_index][N_index][C_index] - eta * weights_derivatives_layers[L_index][N_index][C_index])
                })
            })
        })

        biases_layers.forEach((placeholder, index) => {

            let L_index = index

            biases_layers[L_index].forEach((placeholder, index) => {

                let N_index = index
                new_biases_layer[L_index][N_index] = (biases_layers[L_index][N_index] - eta * biases_derivatives_layers[L_index][N_index])

            })

        })
        return(every_layer_neural_training(new_weights_layers, new_biases_layer, inputs, target_values, iterations - 1))
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
every_layer_neural_training(weights_layers, biases_layers, L0_inputs, real_values, 50000)
console.timeEnd()

console.log("\x1b[31m%s\x1b[0m","real values: ", real_values)
console.log(results)