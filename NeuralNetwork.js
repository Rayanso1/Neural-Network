const real_values = [Math.random(),Math.random(),Math.random(),Math.random(),Math.random()]

const L0_inputs = [0.5, 0.5, -1, -1, -0.5]

const L1_biases = [0, 0, 0, 0, 0]

const L2_biases = [0, 0, 0, 0, 0]

const L1_weights = [

    [-0.2, 0.92, -0.5, 0.2, -0.3],
    [0.3, 0.3, -0.3, -0.23, 0.2],
    [0.2, -0.2, -1, 1, -0.4],
    [0, 0, 0, 0, 0],
    [-0.9, 0.1, -0.9, -0.2, 0.1],

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

function single_node_calculator(weights, bias_index, biases=L1_biases, inputs=L0_inputs, n=0, base_sum=0) {

    if (n < weights.length) {

        let intermetiate_sum = (base_sum + (weights[n]*inputs[n]))

        return(single_node_calculator(weights, bias_index, biases, inputs, n+1, intermetiate_sum))
    }

    return(scaling_function(base_sum + biases[bias_index]))

}

function L1_and_L2_neural_training(L1_weights, L2_weights, L1_biases, L2_biases, inputs, target_values, iterations) {

    if (iterations > 0) {
        
        let L1_results = []

        L1_weights.forEach((placeholder, index) => {

            L1_results.push(single_node_calculator(L1_weights[index], index, L1_biases, inputs))

        })

        let L2_results = []

        L2_weights.forEach((placeholder, index) => {

            L2_results.push(single_node_calculator(L2_weights[index], index, L2_biases, L1_results))

        })
        console.log(L2_results)

        let L2_biases_derivatives = []
        let L2_weights_derivatives = [[], [], [], [], []]

        let L1_results_derivatives = []
        let L1_biases_derivatives = []
        let L1_weights_derivatives = [[], [], [], [], []]

        L2_biases.forEach((placeholder, index) => {

            let node_index = index
            let intermediate_derivative = 0

            L2_weights[index].forEach((placeholder, index) => {

                let c_index = index
                intermediate_derivative += eval(scaling_function_derivative((L2_weights[node_index])[c_index] * L1_results[c_index] + L2_biases[node_index])*2*(L2_results[c_index] - target_values[c_index]))

            })
            
            L2_biases_derivatives.push(intermediate_derivative/L2_weights[node_index].length)
        })

        L2_weights.forEach((placeholder, index) => {

            let node_index = index

            L2_weights[index].forEach((placeholder, index) => {

                let c_index = index

                L2_weights_derivatives[node_index].push(L1_results[c_index]  *  scaling_function_derivative((L2_weights[node_index])[c_index] * L1_results[c_index] + L2_biases[node_index] )  *  2*(L2_results[node_index] - target_values[node_index]))

            })
        })

        L2_weights.forEach((placeholder, index) => {

            let intermediate_derivative = 0
            let node_index = index
        
            L2_weights[node_index].forEach((placeholder, index) => {
            
                let c_index = index 
                intermediate_derivative += (eval(L2_weights[node_index][c_index] * scaling_function_derivative(L2_weights[node_index][c_index] * L1_results[c_index] + L1_biases[node_index]) *2*(L2_results[node_index] - target_values[node_index])))
        
            })
        
            L1_results_derivatives.push(intermediate_derivative/L2_weights[node_index].length)
        })
        
        L1_weights.forEach((placeholder, index) => {

            let node_index = index
            let intermediate_derivative = 0

            L1_weights[node_index].forEach((placeholder, index) => {

                let c_index = index

                intermediate_derivative += (eval(scaling_function_derivative((L1_weights[node_index])[c_index] * inputs[c_index] + L1_biases[node_index])*2*(L1_results_derivatives[node_index])))

            })

            L1_biases_derivatives.push(intermediate_derivative/L1_weights[node_index].length)
        })

        L1_weights.forEach((placeholder, index) => {

            let node_index = index

            L1_weights[node_index].forEach((placeholder, index) => {

                let c_index = index

                L1_weights_derivatives[node_index].push(eval(inputs[c_index]  *  scaling_function_derivative((L1_weights[node_index])[c_index] * inputs[c_index] + L1_biases[node_index] )  *  2*(L1_results_derivatives[node_index])))

            })
        })

        let new_L2_weights = [[], [], [], [], []]
        let new_L1_weights = [[], [], [], [], []]
        let new_L2_biases = []
        let new_L1_biases = []

        L2_weights_derivatives.forEach((placeholder, index) => {

            let node_index = index
            L2_weights_derivatives[node_index].forEach((placeholder, index) => {new_L2_weights[node_index].push(L2_weights[node_index][index] - eta * L2_weights_derivatives[node_index][index])})

        })

        L1_weights_derivatives.forEach((placeholder, index) => {

            let node_index = index
            L1_weights_derivatives[node_index].forEach((placeholder, index) => {new_L1_weights[node_index].push(L1_weights[node_index][index] - eta * L1_weights_derivatives[node_index][index])})

        })

        L2_biases_derivatives.forEach((derivative, index) => {new_L2_biases.push(L2_biases[index] - eta * derivative)})

        L1_biases_derivatives.forEach((derivative, index) => {new_L1_biases.push(L1_biases[index] - eta * derivative)})

        return(L1_and_L2_neural_training(new_L1_weights, new_L2_weights, new_L1_biases, new_L2_biases, inputs, target_values, iterations-1))

    }

    return{
        L1_biases: L1_biases,
        L2_biases: L2_biases,

        L1_weights: L1_weights,
        L2_weights: L2_weights,
    }

}

console.time()
let trained_w_and_b_L1_L2 = L1_and_L2_neural_training(L1_weights, L2_weights, L1_biases, L2_biases, L0_inputs, real_values, 5000)
console.timeEnd()

console.log("\x1b[31m%s\x1b[0m","real values: ", real_values)

console.log(trained_w_and_b_L1_L2)