import re
import json

# Raw text -> Latency in seconds
raw_output_small_model = """
mean_inference_latency_batch_1 : 0.0001891994975624284  mean_inference_throughput_batch_1 : 5285.426298080095 samples/s 
mean_inference_latency_batch_2 : 0.0001887076812264807  mean_inference_throughput_batch_2 : 10598.402709424776  
mean_inference_latency_batch_4 : 0.0002623490638133743  mean_inference_throughput_batch_4 : 15246.86210752197  
mean_inference_latency_batch_8 : 0.00019754289956617105  mean_inference_throughput_batch_8 : 40497.53252366449  
mean_inference_latency_batch_16 : 0.00020614718891563216  mean_inference_throughput_batch_16 : 77614.44666872543  
mean_inference_latency_batch_32 : 0.00021897435812425864  mean_inference_throughput_batch_32 : 146135.83194906026  
mean_inference_latency_batch_64 : 0.0002259296896570016  mean_inference_throughput_batch_64 : 283273.96942456975 samples/s 
mean_inference_latency_batch_128 : 0.0002527798657641985  mean_inference_throughput_batch_128 : 506369.4436779338  
mean_inference_latency_batch_256 : 0.0002876289227870122  mean_inference_throughput_batch_256 : 890035.6665089857  
mean_inference_latency_batch_512 : 0.0003536633796092727  mean_inference_throughput_batch_512 : 1447704.3129703098  
mean_inference_latency_batch_1024 : 0.0004828051122695364  mean_inference_throughput_batch_1024 : 2120938.6022994923  
mean_inference_latency_batch_2048 : 0.0007393659721494345  mean_inference_throughput_batch_2048 : 2769940.837344994  
mean_inference_latency_batch_4096 : 0.001121196447242617  mean_inference_throughput_batch_4096 : 3653240.2596114026 samples/s 
mean_inference_latency_batch_8192 : 0.0020157866453001014  mean_inference_throughput_batch_8192 : 4063922.1512356093  
mean_inference_latency_batch_10000 : 0.002381040163689259  mean_inference_throughput_batch_10000 : 4199845.156961016
"""

raw_output_large_model = """
mean_inference_latency_batch_1 : 0.0003774191072473975  mean_inference_throughput_batch_1 : 2649.5743877230407 samples/s 
mean_inference_latency_batch_2 : 0.00041588813222515644  mean_inference_throughput_batch_2 : 4808.9855060284  
mean_inference_latency_batch_4 : 0.00041443515198393016  mean_inference_throughput_batch_4 : 9651.690936089082  
mean_inference_latency_batch_8 : 0.00043458963563929053  mean_inference_throughput_batch_8 : 18408.170246011116  
mean_inference_latency_batch_16 : 0.0004456417722851818  mean_inference_throughput_batch_16 : 35903.27701542538  
mean_inference_latency_batch_32 : 0.00046443439903059557  mean_inference_throughput_batch_32 : 68901.01178291907  
mean_inference_latency_batch_64 : 0.0005006939952910259  mean_inference_throughput_batch_64 : 127822.58345798678 samples/s 
mean_inference_latency_batch_128 : 0.000700332731476629  mean_inference_throughput_batch_128 : 182770.2665418995  
mean_inference_latency_batch_256 : 0.0011294774360057571  mean_inference_throughput_batch_256 : 226653.48756794032  
mean_inference_latency_batch_512 : 0.001590380493883063  mean_inference_throughput_batch_512 : 321935.53804844775  
mean_inference_latency_batch_1024 : 0.002631922666939141  mean_inference_throughput_batch_1024 : 389069.18233691336  
mean_inference_latency_batch_2048 : 0.0048045325653715285  mean_inference_throughput_batch_2048 : 426264.1520552647  
mean_inference_latency_batch_4096 : 0.009363168197152502  mean_inference_throughput_batch_4096 : 437458.7654257522 samples/s 
mean_inference_latency_batch_8192 : 0.01832511911841587  mean_inference_throughput_batch_8192 : 447036.6575553351  
mean_inference_latency_batch_10000 : 0.021811346733133206  mean_inference_throughput_batch_10000 : 458476.96258063643
"""

# Regular expression to extract batch size and latency
pattern = r"mean_inference_latency_batch_(\d+) : ([0-9.eE+-]+)"

output_json = {}
for model_name, raw_output in zip(['RM-S', 'RM-L'], [raw_output_small_model, raw_output_large_model]):
	# Parse
	latencies = {}
	for match in re.finditer(pattern, raw_output):
		batch_size = match.group(1)  # keep batch size as string for JSON
		latency = float(match.group(2))
		latencies[batch_size] = latency

	# Wrap under model name
	output_json[model_name] = latencies

# Save to JSON
with open('latency_results.json', 'w') as f:
    json.dump(output_json, f, indent=4)

print("Saved to latency_results.json")
