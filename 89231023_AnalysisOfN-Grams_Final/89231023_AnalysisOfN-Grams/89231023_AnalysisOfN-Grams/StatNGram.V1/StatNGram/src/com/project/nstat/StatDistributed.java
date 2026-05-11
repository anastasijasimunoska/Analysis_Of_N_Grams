package com.project.nstat;

import mpi.*;
import java.io.*;
import java.nio.*;
import java.nio.channels.*;
import java.nio.file.*;
import java.util.*;

public class StatDistributed {

    private static final int MIN_BUF = 256 * 1024, MAX_BUF = 2 * 1024 * 1024;
    private static final int MAX_NGRAM = 100, MAX_WORD = 64, BUCKETS = 256;
    private static final long OVERLAP = 8192L;

    public static void run(int n, String filePath) throws Exception {
        n = Math.min(Math.max(n, 1), MAX_NGRAM);
        int rank = MPI.COMM_WORLD.Rank(), size = MPI.COMM_WORLD.Size();
        long startTime = System.nanoTime();

        File file = new File(filePath);
        long fileSize = file.length();
        int bufSize = calcBufSize(fileSize, n, size);
        long[] seg = calcSegment(rank, size, fileSize);

        String workDir = prepareWorkDir(filePath, n, size);
        long localCycles = processSegment(filePath, fileSize, seg, n, bufSize, workDir, rank, size);

        MPI.COMM_WORLD.Barrier();
        long localUnique = countUniqueSignatures(workDir, rank, size);
        MPI.COMM_WORLD.Barrier();

        long totalCycles = reduceSum(localCycles, rank, size, 100);
        long totalUnique = reduceSum(localUnique, rank, size, 101);

        if (rank == 0) {
            printResults(totalUnique, totalCycles, startTime, n, size);
            cleanup(workDir);
        }
    }


    private static int calcBufSize(long fileSize, int n, int size) {
        int scaled = (int) (MIN_BUF * (1 + Math.log(n) / Math.log(2)));
        long perProc = Math.max(1, fileSize / Math.max(1, size));
        return (int) Math.min(Math.max(scaled, MIN_BUF), Math.min(MAX_BUF, perProc));
    }

    private static long[] calcSegment(int rank, int size, long fileSize) {
        long segSize = fileSize / size;
        long ownedStart = rank * segSize;
        long ownedEnd = (rank == size - 1) ? fileSize : (rank + 1) * segSize;
        return new long[]{
                ownedStart, ownedEnd,
                Math.max(0, ownedStart - OVERLAP),
                Math.min(fileSize, ownedEnd + OVERLAP)
        };
    }



    private static File ownerFile(String dir, int src, int owner) {
        return new File(dir, "sig_" + src + "_to_" + owner + ".bin");
    }

    private static File bucketFile(String dir, int owner, int bucket) {
        return new File(dir, "bucket_" + owner + "_" + bucket + ".bin");
    }

    private static String prepareWorkDir(String filePath, int n, int size) throws IOException {
        File src = new File(filePath);
        String key = src.getAbsolutePath() + "|" + src.length() + "|" + src.lastModified() + "|" + n + "|" + size;
        File dir = new File(System.getProperty("java.io.tmpdir"), "nstat_mpj_" + Integer.toHexString(key.hashCode()));
        if (!dir.mkdirs() && !dir.isDirectory())
            throw new IOException("Failed to create temp dir: " + dir);
        return dir.getAbsolutePath();
    }


    private static long processSegment(String filePath, long fileSize, long[] seg,
                                       int n, int bufSize, String workDir, int rank, int size) throws Exception {
        long ownedStart = seg[0], ownedEnd = seg[1], readStart = seg[2], readEnd = seg[3];
        DataOutputStream[] outs = new DataOutputStream[size];
        try {
            for (int i = 0; i < size; i++)
                outs[i] = bufferedOutput(ownerFile(workDir, rank, i));

            try (FileChannel ch = FileChannel.open(Paths.get(filePath), StandardOpenOption.READ)) {
                ArrayDeque<long[]> window = new ArrayDeque<>(MAX_NGRAM);
                StringBuilder word = new StringBuilder(32);
                boolean skipFirst = readStart > 0, inWord = false;
                long wordStart = -1, cycles = 0, pos = readStart;
                ByteBuffer buf = ByteBuffer.allocateDirect(bufSize);

                while (pos < readEnd) {
                    buf.clear();
                    buf.limit((int) Math.min(bufSize, readEnd - pos));
                    int read = ch.read(buf, pos);
                    if (read <= 0) break;
                    buf.flip();

                    while (buf.hasRemaining()) {
                        char c = (char) (buf.get() & 0xFF);
                        if (Character.isWhitespace(c)) {
                            if (skipFirst) { skipFirst = false; }
                            if (inWord) {

                                if (!word.isEmpty()) {
                                    long h1 = hash1(word), h2 = hash2(word);
                                    word.setLength(0);
                                    window.addLast(new long[]{h1, h2, wordStart});
                                    if (window.size() > n) window.pollFirst();
                                    if (window.size() == n) {
                                        long[] first = window.peekFirst();
                                        if (first[2] >= ownedStart && first[2] < ownedEnd) {
                                            long[] sig = buildSig(window);
                                            int owner = Math.floorMod(sigHash(sig), size);
                                            outs[owner].writeLong(sig[0]);
                                            outs[owner].writeLong(sig[1]);
                                            cycles++;
                                        }
                                    }
                                }
                                inWord = false;
                            }
                        } else {
                            if (!skipFirst) {
                                if (!inWord) { inWord = true; wordStart = pos; }
                                if (word.length() < MAX_WORD) word.append(Character.toLowerCase(c));
                            }
                        }
                        pos++;
                    }
                }


                if (inWord && readEnd >= fileSize && !word.isEmpty()) {
                    long h1 = hash1(word), h2 = hash2(word);
                    window.addLast(new long[]{h1, h2, wordStart});
                    if (window.size() > n) window.pollFirst();
                    if (window.size() == n) {
                        long[] first = window.peekFirst();
                        if (first[2] >= ownedStart && first[2] < ownedEnd) {
                            long[] sig = buildSig(window);
                            int owner = Math.floorMod(sigHash(sig), size);
                            outs[owner].writeLong(sig[0]);
                            outs[owner].writeLong(sig[1]);
                            cycles++;
                        }
                    }
                }

                return cycles;
            }
        } finally {
            for (DataOutputStream o : outs) if (o != null) o.close();
        }
    }

    private static long[] buildSig(ArrayDeque<long[]> window) {
        long a = 0x9E3779B97F4A7C15L, b = 0xC2B2AE3D27D4EB4FL;
        for (long[] t : window) {
            a = mix64(a ^ t[0] ^ Long.rotateLeft(t[1], 13));
            b = mix64(b ^ t[1] ^ Long.rotateLeft(t[0], 29));
        }
        return new long[]{a, b};
    }


    private static long countUniqueSignatures(String workDir, int rank, int size) throws IOException {
        DataOutputStream[] buckets = new DataOutputStream[BUCKETS];
        try {
            for (int i = 0; i < BUCKETS; i++)
                buckets[i] = bufferedOutput(bucketFile(workDir, rank, i));

            for (int src = 0; src < size; src++) {
                File f = ownerFile(workDir, src, rank);
                if (!f.exists() || f.length() == 0) continue;
                try (DataInputStream in = bufferedInput(f)) {
                    while (true) {
                        try {
                            long a = in.readLong(), b = in.readLong();
                            int bucket = Math.floorMod(sigHash(new long[]{a, b}), BUCKETS);
                            buckets[bucket].writeLong(a);
                            buckets[bucket].writeLong(b);
                        } catch (EOFException e) { break; }
                    }
                }
            }
        } finally {
            for (DataOutputStream o : buckets) if (o != null) o.close();
        }

        long count = 0;
        for (int i = 0; i < BUCKETS; i++) {
            File f = bucketFile(workDir, rank, i);
            if (!f.exists() || f.length() == 0) continue;
            Set<Long> seen = new HashSet<>();
            try (DataInputStream in = bufferedInput(f)) {
                while (true) {
                    try { seen.add(in.readLong() ^ in.readLong()); }
                    catch (EOFException e) { break; }
                }
            }
            count += seen.size();
            if (!f.delete()) f.deleteOnExit();
        }
        return count;
    }


    private static long reduceSum(long val, int rank, int size, int tag) throws Exception {
        if (rank == 0) {
            long[] buf = new long[1];
            for (int src = 1; src < size; src++) {
                MPI.COMM_WORLD.Recv(buf, 0, 1, MPI.LONG, src, tag);
                val += buf[0];
            }
            return val;
        } else {
            MPI.COMM_WORLD.Send(new long[]{val}, 0, 1, MPI.LONG, 0, tag);
            return 0L;
        }
    }


    private static long hash1(CharSequence s) {
        long h = 0xcbf29ce484222325L;
        for (int i = 0; i < s.length(); i++) { h ^= s.charAt(i); h *= 0x100000001b3L; }
        return mix64(h);
    }

    private static long hash2(CharSequence s) {
        long h = 0x9e3779b97f4a7c15L;
        for (int i = 0; i < s.length(); i++)
            h ^= (long) s.charAt(i) + 0x9e3779b97f4a7c15L + (h << 6) + (h >>> 2);
        return mix64(h);
    }

    private static long mix64(long z) {
        z = (z ^ (z >>> 33)) * 0xff51afd7ed558ccdL;
        z = (z ^ (z >>> 33)) * 0xc4ceb9fe1a85ec53L;
        return z ^ (z >>> 33);
    }

    private static int sigHash(long[] sig) {
        long x = sig[0] * 0x9E3779B97F4A7C15L ^ Long.rotateLeft(sig[1], 17);
        x ^= (x >>> 32);
        return (int) x;
    }


    private static DataOutputStream bufferedOutput(File f) throws IOException {
        return new DataOutputStream(new BufferedOutputStream(new FileOutputStream(f), 1024 * 1024));
    }

    private static DataInputStream bufferedInput(File f) throws IOException {
        return new DataInputStream(new BufferedInputStream(new FileInputStream(f), 1024 * 1024));
    }


    private static void printResults(long unique, long cycles,
                                     long startTime, int n, int size) {
        long ms = System.nanoTime() - startTime;
        long mem = (Runtime.getRuntime().totalMemory()
                - Runtime.getRuntime().freeMemory()) / (1024 * 1024);

        System.out.println("\n=== N-Gram Analysis Results ===");
        System.out.printf("N-gram size: %d | Processes: %d | Memory: %,d MB%n",
                n, size, mem);
        System.out.printf("Time: %.3f s%n", ms / 1e9);
        System.out.printf("Unique %d-grams: %,d%n", n, unique);
        System.out.printf("Cycle count:    %,d%n", cycles);

        System.out.println("\n=== P(B|A) Statistical Summary ===");
        System.out.printf("Total n-grams formed:          %,d%n", cycles);
        System.out.printf("Total unique n-grams:          %,d%n", unique);

        if (cycles > 0) {
            double avgProb = (double) unique / cycles;
            long repeated = cycles - unique;
            double repetitionRate = ((double) repeated / cycles) * 100;

            System.out.printf("Average P(B|A):                %.4f%n", avgProb);
            System.out.printf("Repeated n-grams (duplicates): %,d%n", repeated);
            System.out.printf("Repetition rate:               %.2f%%%n",
                    repetitionRate);
        }
    }

    private static void cleanup(String workDir) {
        File dir = new File(workDir);
        File[] files = dir.listFiles();
        if (files != null) for (File f : files) { if (!f.delete()) f.deleteOnExit(); }
        if (!dir.delete()) dir.deleteOnExit();
    }
}