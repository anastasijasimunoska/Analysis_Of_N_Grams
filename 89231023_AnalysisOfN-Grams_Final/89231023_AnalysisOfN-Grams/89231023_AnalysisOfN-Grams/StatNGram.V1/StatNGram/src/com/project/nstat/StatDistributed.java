package com.project.nstat;

import mpi.*;
import java.io.*;
import java.nio.*;
import java.nio.channels.*;
import java.nio.file.*;
import java.util.*;

public class StatDistributed {

    private static final int MIN_BUF    = 256 * 1024;
    private static final int MAX_BUF    = 2 * 1024 * 1024;
    private static final int MAX_NGRAM  = 100;
    private static final int MAX_WORD   = 64;
    private static final long OVERLAP   = 8192L;
    private static final int CHUNK_SIZE = 8192;

    public static void run(int n, String filePath) throws Exception {
        n = Math.min(Math.max(n, 1), MAX_NGRAM);
        int rank = MPI.COMM_WORLD.Rank();
        int size = MPI.COMM_WORLD.Size();
        long startTime = System.nanoTime();

        File file = new File(filePath);
        long fileSize = file.length();
        int bufSize = calcBufSize(fileSize, n, size);
        long[] seg = calcSegment(rank, size, fileSize);


        @SuppressWarnings("unchecked")
        List<long[]>[] outChunks = new List[size];
        int[] outPos   = new int[size];
        int[] outCount = new int[size];

        for (int i = 0; i < size; i++) {
            outChunks[i] = new ArrayList<>();
            outChunks[i].add(new long[CHUNK_SIZE * 2]);
            outPos[i] = 0;
        }

        long localCycles = singlePass(filePath, fileSize, seg, n,
                bufSize, rank, size, outChunks, outPos, outCount);


        long[][] sendArrs = new long[size][];
        for (int i = 0; i < size; i++) {
            sendArrs[i] = flatten(outChunks[i], outPos[i], outCount[i]);
            outChunks[i] = null;
        }


        Set<Long> seen = new HashSet<>();
        long[] own = sendArrs[rank];
        for (int i = 0; i + 1 < own.length; i += 2)
            seen.add(own[i] ^ own[i + 1]);
        sendArrs[rank] = null;


        MPI.COMM_WORLD.Barrier();


        int[] mySendSizes = new int[size];
        for (int i = 0; i < size; i++)
            mySendSizes[i] = sendArrs[i] == null ? 0 : sendArrs[i].length;


        int[] myRecvSizes = new int[size];
        MPI.COMM_WORLD.Alltoall(
                mySendSizes, 0, 1, MPI.INT,
                myRecvSizes, 0, 1, MPI.INT);


        MPI.COMM_WORLD.Barrier();


        for (int sender = 0; sender < size; sender++) {
            for (int receiver = 0; receiver < size; receiver++) {
                if (sender == receiver) continue;

                // Only the relevant pair participates
                if (rank == sender) {
                    long[] sendArr = sendArrs[receiver] != null ?
                            sendArrs[receiver] : new long[0];
                    if (sendArr.length > 0) {
                        MPI.COMM_WORLD.Send(
                                sendArr, 0, sendArr.length,
                                MPI.LONG, receiver, 201);
                    }
                    sendArrs[receiver] = null;

                } else if (rank == receiver) {
                    int recvLen = myRecvSizes[sender];
                    if (recvLen > 0) {
                        long[] recvArr = new long[recvLen];
                        MPI.COMM_WORLD.Recv(
                                recvArr, 0, recvLen,
                                MPI.LONG, sender, 201);
                        for (int j = 0; j + 1 < recvArr.length; j += 2)
                            seen.add(recvArr[j] ^ recvArr[j + 1]);
                        recvArr = null;
                    }
                }
            }
        }

        long localUnique = seen.size();

        MPI.COMM_WORLD.Barrier();

        long totalCycles = reduceSum(localCycles, rank, size, 100);
        long totalUnique  = reduceSum(localUnique, rank, size, 101);

        if (rank == 0) {
            printResults(totalUnique, totalCycles, startTime, n, size);
        }
    }


    private static long singlePass(String filePath, long fileSize,
                                   long[] seg, int n, int bufSize, int rank, int size,
                                   List<long[]>[] outChunks, int[] outPos, int[] outCount)
            throws Exception {

        long ownedStart = seg[0], ownedEnd = seg[1];
        long readStart  = seg[2], readEnd  = seg[3];
        long cycles = 0;

        try (FileChannel ch = FileChannel.open(
                Paths.get(filePath), StandardOpenOption.READ)) {

            ArrayDeque<long[]> window = new ArrayDeque<>(MAX_NGRAM);
            StringBuilder word = new StringBuilder(32);
            boolean skipFirst = readStart > 0, inWord = false;
            long wordStart = -1, pos = readStart;
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
                        if (inWord && !word.isEmpty()) {
                            long h1 = hash1(word), h2 = hash2(word);
                            word.setLength(0);
                            window.addLast(
                                    new long[]{h1, h2, wordStart});
                            if (window.size() > n) window.pollFirst();
                            if (window.size() == n) {
                                long[] first = window.peekFirst();
                                if (first[2] >= ownedStart &&
                                        first[2] < ownedEnd) {
                                    long[] sig = buildSig(window);
                                    int owner = Math.floorMod(
                                            sigHash(sig), size);
                                    appendSig(outChunks[owner],
                                            outPos, outCount,
                                            owner, sig);
                                    cycles++;
                                }
                            }
                            inWord = false;
                        }
                    } else {
                        if (!skipFirst) {
                            if (!inWord) {
                                inWord = true;
                                wordStart = pos;
                            }
                            if (word.length() < MAX_WORD)
                                word.append(
                                        Character.toLowerCase(c));
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
                    if (first[2] >= ownedStart &&
                            first[2] < ownedEnd) {
                        long[] sig = buildSig(window);
                        int owner = Math.floorMod(
                                sigHash(sig), size);
                        appendSig(outChunks[owner],
                                outPos, outCount, owner, sig);
                        cycles++;
                    }
                }
            }
        }

        return cycles;
    }

    private static void appendSig(List<long[]> chunks,
                                  int[] outPos, int[] outCount,
                                  int owner, long[] sig) {
        long[] current = chunks.get(chunks.size() - 1);
        if (outPos[owner] >= current.length) {
            current = new long[CHUNK_SIZE * 2];
            chunks.add(current);
            outPos[owner] = 0;
        }
        current[outPos[owner]++] = sig[0];
        current[outPos[owner]++] = sig[1];
        outCount[owner] += 2;
    }

    private static long[] flatten(List<long[]> chunks,
                                  int lastPos, int totalCount) {
        if (totalCount == 0) return new long[0];
        long[] result = new long[totalCount];
        int dst = 0;
        for (int c = 0; c < chunks.size(); c++) {
            long[] chunk = chunks.get(c);
            int len = (c == chunks.size() - 1) ?
                    lastPos : chunk.length;
            System.arraycopy(chunk, 0, result, dst, len);
            dst += len;
        }
        return result;
    }


    private static int calcBufSize(long fileSize, int n, int size) {
        int scaled = (int) (MIN_BUF *
                (1 + Math.log(n) / Math.log(2)));
        long perProc = Math.max(1,
                fileSize / Math.max(1, size));
        return (int) Math.min(
                Math.max(scaled, MIN_BUF),
                Math.min(MAX_BUF, perProc));
    }

    private static long[] calcSegment(int rank, int size,
                                      long fileSize) {
        long segSize    = fileSize / size;
        long ownedStart = rank * segSize;
        long ownedEnd   = (rank == size - 1) ? fileSize
                : (rank + 1) * segSize;
        return new long[]{
                ownedStart, ownedEnd,
                Math.max(0, ownedStart - OVERLAP),
                Math.min(fileSize, ownedEnd + OVERLAP)
        };
    }


    private static long[] buildSig(ArrayDeque<long[]> window) {
        long a = 0x9E3779B97F4A7C15L, b = 0xC2B2AE3D27D4EB4FL;
        for (long[] t : window) {
            a = mix64(a ^ t[0] ^ Long.rotateLeft(t[1], 13));
            b = mix64(b ^ t[1] ^ Long.rotateLeft(t[0], 29));
        }
        return new long[]{a, b};
    }


    private static long reduceSum(long val, int rank, int size,
                                  int tag) throws Exception {
        if (rank == 0) {
            long[] buf = new long[1];
            for (int src = 1; src < size; src++) {
                MPI.COMM_WORLD.Recv(buf, 0, 1,
                        MPI.LONG, src, tag);
                val += buf[0];
            }
            return val;
        } else {
            MPI.COMM_WORLD.Send(new long[]{val}, 0, 1,
                    MPI.LONG, 0, tag);
            return 0L;
        }
    }


    private static long hash1(CharSequence s) {
        long h = 0xcbf29ce484222325L;
        for (int i = 0; i < s.length(); i++) {
            h ^= s.charAt(i);
            h *= 0x100000001b3L;
        }
        return mix64(h);
    }

    private static long hash2(CharSequence s) {
        long h = 0x9e3779b97f4a7c15L;
        for (int i = 0; i < s.length(); i++)
            h ^= (long) s.charAt(i) + 0x9e3779b97f4a7c15L
                    + (h << 6) + (h >>> 2);
        return mix64(h);
    }

    private static long mix64(long z) {
        z = (z ^ (z >>> 33)) * 0xff51afd7ed558ccdL;
        z = (z ^ (z >>> 33)) * 0xc4ceb9fe1a85ec53L;
        return z ^ (z >>> 33);
    }

    private static int sigHash(long[] sig) {
        long x = sig[0] * 0x9E3779B97F4A7C15L
                ^ Long.rotateLeft(sig[1], 17);
        x ^= (x >>> 32);
        return (int) x;
    }


    private static void printResults(long unique, long cycles,
                                     long startTime, int n, int size) {
        long elapsed = System.nanoTime() - startTime;
        long mem = (Runtime.getRuntime().totalMemory()
                - Runtime.getRuntime().freeMemory())
                / (1024 * 1024);

        System.out.println("\n=== N-Gram Analysis Results ===");
        System.out.printf(
                "N-gram size: %d | Processes: %d | Memory: %,d MB%n",
                n, size, mem);
        System.out.printf("Time: %.3f s%n", elapsed / 1e9);
        System.out.printf("Unique %d-grams: %,d%n", n, unique);
        System.out.printf("Cycle count:    %,d%n", cycles);

        System.out.println("\n=== P(B|A) Statistical Summary ===");
        System.out.printf("Total n-grams formed:          %,d%n", cycles);
        System.out.printf("Total unique n-grams:          %,d%n", unique);

        if (cycles > 0) {
            double avgProb = (double) unique / cycles;
            long repeated  = cycles - unique;
            double repetitionRate =
                    ((double) repeated / cycles) * 100;

            System.out.printf("Average P(B|A):                %.4f%n", avgProb);
            System.out.printf("Repeated n-grams (duplicates): %,d%n", repeated);
            System.out.printf("Repetition rate:               %.2f%%%n", repetitionRate);
        }
    }
}
