using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NeuroNet_17

{
    class Program
    {
        static void Main(string[] args)

        {
            // Значения функции для обучающих выборок
            double[,] func =
            {
               { 0.1, 0.1, 0.99995 },
               { 0.2, 0.2, 0.999200107 },
               { 0.3, 0.3, 0.995952733 },
               { 0.4, 0.4, 0.987227283 },
               { 0.5, 0.5, 0.968912422 },
               { 0.6, 0.6, 0.935896824 },
               { 0.7, 0.7, 0.882332859 },
               { 0.8, 0.8, 0.802095758 },
               { 0.9, 0.9, 0.689498433 },
               { 1.0, 1.0, 0.540302306 }
           };

            // Значения функции для тестовой выборки
            double[] tstX = { 0.45 };
            double[] tstY = { 0.45 };
            double[] tstZ = { 0.979566842 };

            int h_neuron_num = 10; // кол-во нейронов скрытого слоя
            int epoch_num = 100000; // кол-во эпох
            double err_level = 0.0000001; // точность вычислений
            int study_set_num = func.Length / 3; // кол-во обучающих выборок
            int varNum = 17;
            string FIO = "Свиридова Е.К.";
            string grName = "ИС50-2-22";

            // выполняем нормализацию значений
            Vector[] X1 = new Vector[study_set_num];
            Vector[] Y1 = new Vector[study_set_num];
            for (int i = 0; i < study_set_num; i++) 

            {
                X1[i] = new Vector(func[i, 0], func[i, 1]);
                Y1[i] = new Vector(Math.Cos(func[i, 0] * func[i, 1]));
            };

            Vector[] X2 = { new Vector(tstX[0], tstY[0]) };
            Vector[] Y2 = { new Vector(Math.Cos(tstX[0] * tstY[0])) };

            string outputFile = $"NeuroNet_{varNum}.txt"; // Имя выходного файла

            string outString = "ИСиТ. Пр.р. №3. \"Простейшая нейросеть\"\n" +
            "Выполнил : " + FIO + "\n" +
            "Группа : " + grName + "\n" +
            "Вариант №: " + varNum.ToString() + "\n" +
            "------------------------------------\n\n" +
            "Параметры:\n" +
            "- Количество обучающих выборок: " + study_set_num.ToString() + "\n" +
            "- Количество нейронов в скрытом слое: " + h_neuron_num.ToString() + "\n" +
            "- Количество эпох: " + epoch_num.ToString() + "\n" +
            "- Ошибка: " + err_level.ToString("F8") + "\n";

            DateTime startTime = DateTime.Now; // Запоминаем время начала выполнения программы


            outString += "\nИсходные данные для обучающих выборок" + "\n-------------------------------------" +
            "\nЗначения X:\tЗначения Y:\tЗначения Z:";

            for (int i = 0; i < study_set_num; i++)
            {
                outString += "\n" + func[i, 0].ToString("0.000000");
                outString += "\t" + func[i, 1].ToString("0.000000");
                outString += "\t" + func[i, 2].ToString("0.000000");
            }

            outString += "\n\nИсходные данные для тестовой выборки" + "\n------------------------------------" +
            "\nТестовое значение X = " + tstX[0].ToString("0.000000") +
            "\nТестовое значение Y = " + tstY[0].ToString("0.000000") +
            "\nТестовое значение Z = " + tstZ[0].ToString("0.000000") + "\n";

            //вывод данных на экран и запись в файл
            Console.WriteLine(outString);

            // создаём сеть с двумя входами, N нейронами в скрытом слое и одним выходом
            Network network = new Network(new int[] { 2, h_neuron_num, 1 });
            outString = "Обучение нейросети\n" + "------------------";
            Console.WriteLine(outString);

            // запускаем обучение сети
            network.Train(X1, Y1, 0.5, err_level, epoch_num);
            outString = "\n\nОбучающие выборки и результаты расчета:\n" + "--------------------------------------";
            Console.WriteLine(outString);

            for (int i = 0; i < study_set_num; i++)

            {
                {
                    // запускаем сеть с обучающими выборками
                    Vector output = network.Forward(X1[i]);
                    // выводим результаты расчетов на каждой обучающей выборке
                    Console.WriteLine("X1: {0:0.0000000} {1:0.0000000}\t Y1: {2:0.0000000}\t результат: {3:0.0000000}", X1[i][0], X1[i][1], Math.Cos(X1[i][0] * X1[i][1]), output[0]);
                }

                // запускаем сеть с тестовой выборкой
                Vector output2 = network.Forward(X2[0]);
                // выводим результаты расчетов на тестовой выборке
                outString = "\nТестовая выборка и результат расчета:\n" +
                "-------------------------------------";
                Console.WriteLine(outString);
                Console.WriteLine("X2: {0:0.0000000} {1:0.0000000}\t Y2: {2:0.0000000}\t результат: {3:0.0000000}", X2[0][0], X2[0][1], Math.Cos(X2[0][0] * X2[0][1]), output2[0]);
                outString = "\nИтоги:" + "\n------" + "\nВычисленное нейросетью значение функции Z = " + Math.Cos(X2[0][0] * X2[0][1]).ToString("0.000000") + "\nРазность между тестовым и рассчитанным значениями Z = " +

                Math.Abs(Math.Cos(X2[0][0] * X2[0][1]) - output2[0]).ToString("0.000000");
                TimeSpan executionTime = DateTime.Now - startTime;
                Console.WriteLine(outString);

                Console.WriteLine($"\nЗатраченное время: {executionTime.TotalSeconds:F6} секунд");
                Console.WriteLine("\nНажмите любую клавишу...");
                Console.ReadKey();

            }// End Main


        }

    }
}