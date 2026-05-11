package com.project.nstat;

import java.io.*;
import java.util.*;

public class StatSequential {
    private final Map<String, Integer> nGramFrequencies = new HashMap<>();
    private final Map<String, Integer> startingWordCounts = new HashMap<>();
    private long cycles = 0;

    public Map<String, Integer> getNGramFrequencies() {
        return nGramFrequencies;
    }

    public Map<String, Integer> getStartingWordCounts() {
        return startingWordCounts;
    }

    public long getCycles() {
        return cycles;
    }

    public void processFileAndGenerateNGrams(File file, int n) {
        try (BufferedReader reader = new BufferedReader(
                new FileReader(file), 1024 * 1024)) {
            ArrayDeque<String> window = new ArrayDeque<>();
            String line;

            while ((line = reader.readLine()) != null) {
                String[] words = line.trim()
                        .toLowerCase(Locale.ROOT)
                        .split("\\s+");

                for (String word : words) {
                    if (word.isEmpty()) continue;

                    window.addLast(word);

                    if (window.size() == n) {
                        String nGram = String.join(" ", window);
                        nGramFrequencies.merge(nGram, 1, Integer::sum);
                        startingWordCounts.merge(
                                window.peekFirst(), 1, Integer::sum);
                        cycles++;
                        window.removeFirst();
                    }
                }
            }
        } catch (IOException ex) {
            ex.printStackTrace();
        }
    }

    public static void run(int nGramSize, String filePath) {
        StatSequential stat = new StatSequential();


        long startTime = System.nanoTime();

        stat.processFileAndGenerateNGrams(
                new File(filePath), nGramSize);


        long endTime = System.nanoTime();
        double elapsedSeconds =
                (endTime - startTime) / 1_000_000_000.0;


        System.out.printf("Total unique n-grams: %d%n",
                stat.getNGramFrequencies().size());
        System.out.printf("Total Cycles: %d%n",
                stat.getCycles());
        System.out.printf(
                "Execution time in sequential mode: %.3f seconds%n",
                elapsedSeconds);


        System.out.println(
                "\n--- Top 20 Relative Frequencies P(B|A) ---");
        stat.getNGramFrequencies().entrySet().stream()
                .sorted((a, b) ->
                        b.getValue().compareTo(a.getValue()))
                .limit(20)
                .forEach(e -> {
                    String firstWord = e.getKey().split(" ")[0];
                    int total = stat.getStartingWordCounts()
                            .getOrDefault(firstWord, 1);
                    double prob = (double) e.getValue() / total;
                    System.out.printf("%-40s → %.4f%n",
                            e.getKey(), prob);

                });
    }
}