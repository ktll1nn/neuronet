using NeuroNet_17;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NeuroNet_17
{
    class Network
    {
        struct LayerT
        {
            public Vector x; // вход слоя
            public Vector z; // активированный выход слоя
            public Vector df; // производная функции активации слоя
        }
        Matrix[] weights; // матрицы весов слоя
        LayerT[] L; // значения на каждом слое
        Vector[] deltas; // дельты ошибки на каждом слое
        int layersN; // число слоёв
        public Network(int[] sizes)
        {
            // создаём генератор случайных чисел
            Random random = new Random(DateTime.Now.Millisecond);
            layersN = sizes.Length - 1; // запоминаем число слоёв
            weights = new Matrix[layersN]; // создаём массив матриц весовых коэффициентов
            L = new LayerT[layersN]; // создаём массив значений на каждом слое
            deltas = new Vector[layersN]; // создаём массив для дельт
            for (int k = 1; k < sizes.Length; k++)
            {
                // создаём матрицу весовых коэффициентов
                weights[k - 1] = new Matrix(sizes[k], sizes[k - 1], random);
                L[k - 1].x = new Vector(sizes[k - 1]); // создаём вектор для входа слоя
                L[k - 1].z = new Vector(sizes[k]); // создаём вектор для выхода слоя
                L[k - 1].df = new Vector(sizes[k]); // создаём вектор для производной слоя
                deltas[k - 1] = new Vector(sizes[k]); // создаём вектор для дельт
            }
        }
        // прямое распространение
        public Vector Forward(Vector input)
        {
            for (int k = 0; k < layersN; k++)
            {
                if (k == 0)
                {
                    for (int i = 0; i < input.n; i++)
                        L[k].x[i] = input[i];
                }
                else
                {
                    for (int i = 0; i < L[k - 1].z.n; i++)
                        L[k].x[i] = L[k - 1].z[i];
                }
                for (int i = 0; i < weights[k].n; i++)
                {
                    double y = 0;
                    for (int j = 0; j < weights[k].m; j++)
                        y += weights[k][i, j] * L[k].x[j];
                    // активация с помощью сигмоидальной функции
                    L[k].z[i] = 1 / (1 + Math.Exp(-y));
                    L[k].df[i] = L[k].z[i] * (1 - L[k].z[i]);
                }
            }
            return L[layersN - 1].z; // возвращаем результат
        } // End Forward
          // обратное распространение
        void Backward(Vector output, ref double error)
        {
            int last = layersN - 1;
            error = 0; // обнуляем ошибку
            for (int i = 0; i < output.n; i++)
            {
                double e = L[last].z[i] - output[i]; // находим разность значений векторов
                deltas[last][i] = e * L[last].df[i]; // запоминаем дельту
                error += e * e / 2; // прибавляем к ошибке половину квадрата значения
            }
            // вычисляем каждую предыдущую дельту на основе текущей
            // с помощью умножения на транспонированную матрицу
            for (int k = last; k > 0; k--)
            {
                for (int i = 0; i < weights[k].m; i++)
                {
                    deltas[k - 1][i] = 0;
                    for (int j = 0; j < weights[k].n; j++)
                        deltas[k - 1][i] += weights[k][j, i] * deltas[k][j];
                    // умножаем получаемое значение на производную предыдущего слоя
                    deltas[k - 1][i] *= L[k - 1].df[i];
                }
            }
        } // End Backward
          // обновление весовых коэффициентов, alpha - скорость обучения
        void UpdateWeights(double alpha)
        {
            for (int k = 0; k < layersN; k++)
            {
                for (int i = 0; i < weights[k].n; i++)
                {
                    for (int j = 0; j < weights[k].m; j++)
                    {
                        weights[k][i, j] -= alpha * deltas[k][i] * L[k].x[j];
                    }
                }
            }
        } // End UpdateWeights
          // тренировка сети
        public void Train(Vector[] X, Vector[] Y, double alpha, double eps, int epochs)
        {
            int epoch = 1; // номер эпохи
            double error; // ошибка эпохи
            DateTime startTime = DateTime.Now; // затраченное время
            do
            {
                error = 0; // обнуляем ошибку
                           // проходимся по всем элементам обучающего множества
                for (int i = 0; i < X.Length; i++)
                {
                    Forward(X[i]); // прямое распространение сигнала
                    Backward(Y[i], ref error); // обратное распространение ошибки
                    UpdateWeights(alpha); // обновление весовых коэффициентов
                }
                // выводим в консоль номер эпохи и величину ошибки
                Console.CursorLeft = 0;
                Console.Write("Эпоха:{0} Ошибка:{1:0.0000000}", epoch, error);
                epoch++; // увеличиваем номер эпохи
            } while (epoch <= epochs && error > eps);
            TimeSpan trainTime = DateTime.Now - startTime;
            
        } // End Train
    } // End Class Network
}