#include <algorithm>
#include <iostream>
#include <unistd.h>
#include <fstream>
#include <iomanip>
#include <chrono>
#include <string>
#include <vector>
#include <climits>
#include <random>
#include <unistd.h>
#include <mpi.h>
#include "wireroute.h"

void print_stats(const std::vector<std::vector<int>>& occupancy) {
  int max_occupancy = 0;
  long long total_cost = 0;

  for (const auto& row : occupancy) {
    for (const int count : row) {
      max_occupancy = std::max(max_occupancy, count);
      total_cost += count * count;
    }
  }

  std::cout << "Max occupancy: " << max_occupancy << '\n';
  std::cout << "Total cost: " << total_cost << '\n';
}

void write_output(const std::vector<Wire>& wires, const int num_wires, const std::vector<std::vector<int>>& occupancy, const int dim_x, const int dim_y, const int nproc, std::string input_filename) {
  if (std::size(input_filename) >= 4 && input_filename.substr(std::size(input_filename) - 4) == ".txt") {
    input_filename.resize(std::size(input_filename) - 4);
  }

  const std::string occupancy_filename = input_filename + "_occupancy_" + std::to_string(nproc) + ".txt";
  const std::string wires_filename = input_filename + "_wires_" + std::to_string(nproc) + ".txt";

  std::ofstream out_occupancy(occupancy_filename, std::fstream::out);
  if (!out_occupancy) {
    std::cerr << "Unable to open file: " << occupancy_filename << '\n';
    exit(EXIT_FAILURE);
  }

  out_occupancy << dim_x << ' ' << dim_y << '\n';
  for (const auto& row : occupancy) {
    for (const int count : row) {
      out_occupancy << count << ' ';
    }
    out_occupancy << '\n';
  }

  out_occupancy.close();

  std::ofstream out_wires(wires_filename, std::fstream:: out);
  if (!out_wires) {
    std::cerr << "Unable to open file: " << wires_filename << '\n';
    exit(EXIT_FAILURE);
  }

  out_wires << dim_x << ' ' << dim_y << '\n' << num_wires << '\n';

  for (const auto& [start_x, start_y, end_x, end_y, bend1_x, bend1_y] : wires) {
    out_wires << start_x << ' ' << start_y << ' ' << bend1_x << ' ' << bend1_y << ' ';

    if (start_y == bend1_y) {
    // first bend was horizontal

      if (end_x != bend1_x) {
        // two bends

        out_wires << bend1_x << ' ' << end_y << ' ';
      }
    } else if (start_x == bend1_x) {
      // first bend was vertical

      if (end_y != bend1_y) {
        // two bends

        out_wires << end_x << ' ' << bend1_y << ' ';
      }
    }
    out_wires << end_x << ' ' << end_y << '\n';
  }

  out_wires.close();
}
int refOccupancy(std::vector<std::vector<int>>&occupancy , struct Wire route, int dim_x, int dim_y, int flag){
  // If flag == -1, decrement occupancy along route
  // If flag == 1, increment occupancy along route
  // If flag == 0, calculate cost of adding the route
  int bend2_x;
  int bend2_y;
  int start_x = route.start_x;
  int start_y = route.start_y;
  int end_x = route.end_x;
  int end_y = route.end_y;
  int bend1_x = route.bend1_x;
  int bend1_y = route.bend1_y;

  if (bend1_x == start_x) {
    bend2_x = end_x;
    bend2_y = bend1_y;
  }
  else if (bend1_y == start_y) {
    bend2_x = bend1_x;
    bend2_y = end_y;
  }
  else {

    printf("bend1_x = %d, start_x = %d, bend1_y = %d, start_y = %d\n", bend1_x,start_x,bend1_y,start_y);
    return -109823498;
  }


  int cost = 0;

  // START TO BEND 1
  int stepi1 = 1;
  if(start_y > bend1_y)
  {
    stepi1 = -1;
  }
  for (int i = start_y ; i != bend1_y; i += stepi1){
    if (flag == 0){
      cost += occupancy[i][start_x] + 1;
    }
    else {
      occupancy[i ][start_x] += flag;
    }
  }
  int stepi2 = 1;
  if(start_x > bend1_x)
  {
    stepi2 = -1;
  }
  for (int i = start_x; i != bend1_x; i += stepi2 ) {
    if (flag == 0){
      cost += occupancy[start_y ][+ i] + 1;
    }
    else {
      occupancy[start_y ][ i] += flag;
    }
  }


  int stepi3 = 1;
  if(bend1_x > bend2_x)
  {
    stepi3 = -1;
  }
  // BEND 1 TO BEND 2
  for (int i = bend1_x; i !=  bend2_x; i += stepi3) {
    
    if (flag == 0){
      cost += occupancy[bend1_y ][ + i] + 1;
    }
    else {
      occupancy[bend1_y ][ i] += flag;
    }
  }

  int stepi4 = 1;
  if(bend1_y > bend2_y)
  {
    stepi4 = -1;
  }
  
  for (int i = bend1_y; i !=  bend2_y; i += stepi4) {
    
    if (flag == 0){
      cost += occupancy[i ][ bend1_x] + 1;
    }
    else {
      occupancy[i ][ bend1_x] += flag;
    }
  }

  int stepi5 = 1;
  if(bend2_x > end_x)
  {
    stepi5 = -1;
  }


  // BEND 2 TO END
  for (int i = bend2_x ; i != end_x; i += stepi5) {
    if (flag == 0){
      cost += occupancy[end_y][ i] + 1;
    }
    else {
    
      occupancy[end_y ][ i] += flag;
      
      
    }
  }

  int stepi6 = 1;
  if(bend2_y > end_y)
  {
    stepi6 = -1;
  }
  for (int i = bend2_y; i !=  end_y; i +=stepi6) {
    
    if (flag == 0){
      cost += occupancy[i ][ end_x] + 1;
    }
    else {

        occupancy[i ][ end_x] += flag; 
      
      
    }
  }

  // INCLUDE END POINT
  if (flag == 0){
      cost += occupancy[end_y ][ end_x] + 1;
    }
  else {
   
    occupancy[end_y ][end_x] += flag;
    
    
  }
  return cost;

}

// // Credit: https://stackoverflow.com/questions/5901476/sending-and-receiving-2d-array-over-mpi
// int **alloc_2d_int(int rows, int cols) {
//     int *data = (int *)calloc(rows*cols, sizeof(int));
//     int **array= (int **)calloc(rows,sizeof(int*));
//     for (int i=0; i<rows; i++)
//         array[i] = &(data[cols*i]);

//     return array;
// }


int main(int argc, char *argv[]) {
  const auto init_start = std::chrono::steady_clock::now();
  int pid;
  int nproc;

  // Initialize MPI
  MPI_Init(&argc, &argv);
  // Get process rank
  MPI_Comm_rank(MPI_COMM_WORLD, &pid);
  // Get total number of processes specificed at start of run
  MPI_Comm_size(MPI_COMM_WORLD, &nproc);

  std::string input_filename;
  double SA_prob = 0.1;
  int SA_iters = 5;
  char parallel_mode = '\0';
  int batch_size = 16;

  // Read command line arguments
  int opt;
  while ((opt = getopt(argc, argv, "f:p:i:m:b:")) != -1) {
    switch (opt) {
      case 'f':
        input_filename = optarg;
        break;
      case 'p':
        SA_prob = atof(optarg);
        break;
      case 'i':
        SA_iters = atoi(optarg);
        break;
      case 'm':
        parallel_mode = *optarg;
        break;
      case 'b':
        batch_size = atoi(optarg);
        break;
      default:
        if (pid == 0) {
          std::cerr << "Usage: " << argv[0] << " -f input_filename [-p SA_prob] [-i SA_iters] -m parallel_mode -b batch_size\n";
        }

        MPI_Finalize();
        exit(EXIT_FAILURE);
    }
  }

  // Check if required options are provided
  if (empty(input_filename) || SA_iters <= 0 || (parallel_mode != 'A' && parallel_mode != 'W') || batch_size <= 0) {
    if (pid == 0) {
      std::cerr << "Usage: " << argv[0] << " -f input_filename [-p SA_prob] [-i SA_iters] -m parallel_mode -b batch_size\n";
    }

    MPI_Finalize();
    exit(EXIT_FAILURE);
  }

  if (pid == 0) {
    std::cout << "Number of processes: " << nproc << '\n';
    std::cout << "Simulated annealing probability parameter: " << SA_prob << '\n';
    std::cout << "Simulated annealing iterations: " << SA_iters << '\n';
    std::cout << "Input file: " << input_filename << '\n';
    std::cout << "Parallel mode: " << parallel_mode << '\n';
    std::cout << "Batch size: " << batch_size << '\n';
  }

  int dim_x, dim_y, num_wires;
  // std::vector<Wire> wires;
  // std::vector<std::vector<int>> occupancy;

  // if (pid == 0) {
      std::ifstream fin(input_filename);

      if (!fin) {
        std::cerr << "Unable to open file: " << input_filename << ".\n";
        exit(EXIT_FAILURE);
      }

      /* Read the grid dimension and wire information from file */
      fin >> dim_x >> dim_y >> num_wires;

      std::vector<Wire> wires(num_wires);
       //Changed this line below (bugs)
      std::vector<std::vector<int>> occupancy(dim_y, std::vector<int>(dim_x)); 

  for (auto& wire : wires) {
    fin >> wire.start_x >> wire.start_y >> wire.end_x >> wire.end_y;
    wire.bend1_x = wire.start_x;
    wire.bend1_y = wire.start_y;
  }

      
  // }

  /* Initialize any additional data structures needed in the algorithm */

  if (pid == 0) {
    const double init_time = std::chrono::duration_cast<std::chrono::duration<double>>(std::chrono::steady_clock::now() - init_start).count();
    std::cout << "Initialization time (sec): " << std::fixed << std::setprecision(10) << init_time << '\n';
  }

  const auto compute_start = std::chrono::steady_clock::now();

  /** 
   * (TODO)
   * Implement the wire routing algorithm here
   * Feel free to structure the algorithm into different functions
   * Use MPI to parallelize the algorithm. 
   */


  int num_batches = (num_wires + batch_size - 1) / batch_size;


  
  for(int wireIndex = 0; wireIndex < num_wires; wireIndex++)
  {
    struct Wire currWire = wires[wireIndex];
    currWire.bend1_x = currWire.start_x;
    currWire.bend1_y = currWire.end_y;
    wires[wireIndex] = currWire;
    refOccupancy(occupancy, currWire,  dim_x,  dim_y, 1);
  }
          
      
    

   for (int timestep = 1; timestep < SA_iters; timestep++){
    
    for (int batch_ind = 0; batch_ind < num_batches; batch_ind += nproc){
      int send_counts[nproc];
      int disp[nproc];

      int i = batch_ind * batch_size;
      int b = 0;
      while (b < nproc && i < num_wires){
        disp[b] = (i - batch_ind * batch_size) * 2;
        send_counts[b] = std::min(batch_size, num_wires - i);
        i += batch_size;
        b += 1;
      }
      while (b < nproc){
        disp[b] = 0;
        send_counts[b] = 0;
        b += 1;
      }
      struct Wire* local_wires = (struct Wire *)malloc(sizeof(struct Wire) * batch_size);

      int num_local_wires = send_counts[pid];

      int start = (batch_ind + pid) * batch_size;
      for (int wireIndex = start; wireIndex < start + num_local_wires; wireIndex ++ ){
       
        struct Wire currWire = wires[wireIndex];
        int xi, yi, xf, yf;
        xi = currWire.start_x;
        yi = currWire.start_y;
        xf = currWire.end_x;
        yf = currWire.end_y;
         
        int delta_x = std::abs(xf - xi);
        int delta_y = std::abs(yf - yi);
        if(delta_x != 0 && delta_y != 0 ){
          refOccupancy(occupancy,currWire,dim_x,dim_y, -1);
          int initial_cost = refOccupancy(occupancy, currWire, dim_x, dim_y, 0);
          int min_cost = initial_cost;
          struct Wire best_route = currWire;
          struct Wire* possRoutes = (struct Wire*)malloc(sizeof(struct Wire)*(delta_x + delta_y));
          
          for (int d_x = 0; d_x < delta_x; d_x += 1 ){
            if(xi > xf)
            {
              currWire.bend1_x = xi - d_x - 1;
            }
            else {
              currWire.bend1_x = xi + d_x + 1;
            }
            currWire.bend1_y = yi;
            int cost = refOccupancy(occupancy, currWire, dim_x, dim_y, 0);
            if (cost < min_cost) {
              min_cost = cost;
              best_route = currWire;
            }
            possRoutes[d_x] = currWire;
          }

          for (int d_y = 0; d_y < delta_y; d_y += 1) {
            currWire.bend1_x = xi;
            if (yi > yf) {
              currWire.bend1_y = yi - d_y - 1;
            }
            else {
              currWire.bend1_y = yi + d_y + 1;
            }
            int cost = refOccupancy(occupancy, currWire, dim_x, dim_y, 0);
            if (cost < min_cost) {
              min_cost = cost;
              best_route = currWire;
            }
            possRoutes[delta_x + d_y] = currWire;
          }

          std::random_device rd;  // obtain a random number from hardware
          std::mt19937 gen(rd()); // seed the generator

          // Define a distribution (uniform distribution between 0 and 1)
          std::uniform_real_distribution<> dis(0.0, 1.0);

          // Generate a random number between 0 and 1
          float random_number = dis(gen);
          if (random_number < SA_prob){
            std::uniform_int_distribution<> dis(0, delta_x + delta_y - 1);
            int random_index= dis(gen);
            local_wires[wireIndex - start] = possRoutes[random_index];
          }
          else{
            local_wires[wireIndex - start] = best_route;
          }
          free(possRoutes);
          refOccupancy(occupancy,wires[wireIndex],dim_x,dim_y, 1); //added 

        }
        else{
          local_wires[wireIndex - start] = currWire;
        }

      }

      int best_bends[batch_size*2];
      for (int i = 0; i < num_local_wires; i ++) {
        best_bends[2*i] = local_wires[i].bend1_x;
        best_bends[2*i+1] = local_wires[i].bend1_y;
      }

      int wire_tot = 0;
      int recv_counts[nproc];
      for (int i = 0; i < nproc; i ++){
        wire_tot += send_counts[i];
        recv_counts[i] = 2*send_counts[i];
      }
      int *recv_all = (int*)malloc(sizeof(int)* wire_tot*2);
      
      MPI_Allgatherv(best_bends,
                    num_local_wires * 2,
                    MPI_INT,
                    recv_all,
                    recv_counts,
                    disp,
                    MPI_INT,
                    MPI_COMM_WORLD);


        for (int i = (batch_ind * batch_size); i < std::min(num_wires,(batch_ind * batch_size) + wire_tot); i ++){
          
          refOccupancy(occupancy, wires[i], dim_x, dim_y, -1);
        }
      

      for (int i = batch_size * batch_ind ; i < (batch_size * batch_ind ) + wire_tot; i ++) {
        struct Wire cur_wire = wires[i];
        cur_wire.bend1_x = recv_all[2 * (i - (batch_size * batch_ind))];
        cur_wire.bend1_y = recv_all[2 * (i - (batch_size * batch_ind)) + 1];
        wires[i] = cur_wire;
      }

        for (int i = batch_ind * batch_size ; i < std::min(num_wires,(batch_ind * batch_size) + wire_tot); i ++){
          int res = refOccupancy(occupancy, wires[i], dim_x, dim_y, 1);

        }

      free(recv_all);
      free(local_wires);

    }

    
    
    
   }


  if (pid == 0) {
    const double compute_time = std::chrono::duration_cast<std::chrono::duration<double>>(std::chrono::steady_clock::now() - compute_start).count();
    std::cout << "Computation time (sec): " << std::fixed << std::setprecision(10) << compute_time << '\n';
  }

  if (pid == 0) {
    

    print_stats(occupancy);
    write_output(wires, num_wires, occupancy, dim_x, dim_y, nproc, input_filename);
  }

  MPI_Finalize();
}
