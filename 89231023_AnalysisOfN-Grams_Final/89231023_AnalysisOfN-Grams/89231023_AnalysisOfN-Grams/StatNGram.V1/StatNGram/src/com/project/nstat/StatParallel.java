package com.project.nstat;

import java.io.*;
import java.util.*;
import java.util.concurrent.*;

public class StatParallel {

    private static final int NUM_THREADS = Runtime.getRuntime().availableProcessors();
    private static final int CHUNK_WORDS = 200_000;

    private static class PartialResult {
        final Map<String, Integer> freqMap = new HashMap<>();
        final Map<String, Integer> startMap = new HashMap<>();
        long cycles = 0;
    }

    private static class ChunkTask implements Callable<PartialResult> {
        private final List<String> prefix;
        private final List<String> ownedWords;
        private final int n;

        ChunkTask(List<String> prefix, List<String> ownedWords, int n) {
            this.prefix = prefix;
            this.ownedWords = ownedWords;
            this.n = n;
        }

        @Override
        public PartialResult call() {
            PartialResult result = new PartialResult();

            List<String> allWords = new ArrayList<>(prefix.size() + ownedWords.size());
            allWords.addAll(prefix);
            allWords.addAll(ownedWords);

            int prefixSize = prefix.size();
            int lastStart = allWords.size() - n;

            for (int i = prefixSize; i <= lastStart; i++) {
                String ngram = String.join(" ", allWords.subList(i, i + n));
                result.freqMap.merge(ngram, 1, Integer::sum);
                result.startMap.merge(allWords.get(i), 1, Integer::sum);
                result.cycles++;
            }

            return result;
        }
    }

    private static List<String> tailOf(List<String> words, int k) {
        if (k <= 0 || words.isEmpty()) return Collections.emptyList();
        int from = Math.max(0, words.size() - k);
        return new ArrayList<>(words.subList(from, words.size()));
    }

    public static void run(int n, String filePath) {
        ExecutorService executor = Executors.newFixedThreadPool(NUM_THREADS);
        List<Future<PartialResult>> futures = new ArrayList<>();

        try (BufferedReader reader = new BufferedReader(
                new FileReader(filePath), 1024 * 1024)) {
            List<String> currentChunk = new ArrayList<>(CHUNK_WORDS);
            List<String> previousTail = Collections.emptyList();
            String line;

            
            long startTime = System.nanoTime();

            while ((line = reader.readLine()) != null) {
                String[] tokens = line.trim()
                        .toLowerCase(Locale.ROOT)
                        .split("\\s+");

                for (String token : tokens) {
                    if (token.isEmpty()) continue;

                    currentChunk.add(token);

                    if (currentChunk.size() >= CHUNK_WORDS) {
                        List<String> owned = new ArrayList<>(currentChunk);
                        List<String> prefix = new ArrayList<>(previousTail);

                        futures.add(executor.submit(
                                new ChunkTask(prefix, owned, n)));

                        previousTail = tailOf(owned, n - 1);
                        currentChunk.clear();
                    }
                }
            }

            if (!currentChunk.isEmpty()) {
                futures.add(executor.submit(
                        new ChunkTask(
                                new ArrayList<>(previousTail),
                                new ArrayList<>(currentChunk), n)
                ));
            }

            Map<String, Integer> freqMap = new HashMap<>();
            Map<String, Integer> startMap = new HashMap<>();
            long cycles = 0;

            for (Future<PartialResult> future : futures) {
                PartialResult part = future.get();
                part.freqMap.forEach((k, v) ->
                        freqMap.merge(k, v, Integer::sum));
                part.startMap.forEach((k, v) ->
                        startMap.merge(k, v, Integer::sum));
                cycles += part.cycles;
            }

           
            long endTime = System.nanoTime();
            double elapsedSeconds =
                    (endTime - startTime) / 1_000_000_000.0;

          
            System.out.printf("Total unique n-grams: %d%n",
                    freqMap.size());
            System.out.printf("Total Cycles: %d%n", cycles);
            System.out.printf(
                    "Execution time in parallel mode: %.3f seconds%n",
                    elapsedSeconds);

           
            System.out.println(
                    "\n--- Top 20 Relative Frequencies P(B|A) ---");
            freqMap.entrySet().stream()
                    .sorted((a, b) ->
                            b.getValue().compareTo(a.getValue()))
                    .limit(20)
                    .forEach(e -> {
                        String firstWord = e.getKey().split(" ")[0];
                        int total = startMap
                                .getOrDefault(firstWord, 1);
                        double prob = (double) e.getValue() / total;
                        System.out.printf("%-40s → %.4f%n",
                                e.getKey(), prob);
                    });

        } catch (IOException | InterruptedException |
                 ExecutionException e) {
            e.printStackTrace();
        } finally {
            executor.shutdown();
        }
    }
}
