// Taken from the Polyhedral benchmark: https://web.cse.ohio-state.edu/~pouchet.2/software/polybench/

#pragma ACCEL kernel
void kernel_mvt(double x1[120], double x2[120], double y_1[120], double y_2[120], double A[120][120])
{
  int i;
  int j;
  
  for (i = 0; i < 120; i++) {
    for (j = 0; j < 120; j++) {
      x1[i] += A[i][j] * y_1[j];
    }
  }
  
  for (i = 0; i < 120; i++) {
    for (j = 0; j < 120; j++) {
      x2[i] += A[j][i] * y_2[j];
    }
  }
}
