package com.project.nstat;

import com.project.nstat.StatSequential;
import com.project.nstat.StatParallel;

public class MainSeqParl {

    public static void main(String[] args) {
        if (args.length < 3) {
            System.out.println("Usage: java Main <mode> <nGramSize> <inputFilePath>");
            System.out.println("Supported modes: sequential, parallel");
            return;
        }

        String mode = args[0].toLowerCase();
        int nGramSize;
        try {
            nGramSize = Integer.parseInt(args[1]);
        } catch (NumberFormatException e) {
            System.out.println("Invalid n-gram size. Must be an integer.");
            return;
        }

        String filePath = args[2];

        long startTime = System.nanoTime();

        switch (mode) {
            case "sequential":
                StatSequential.run(nGramSize, filePath);
                break;
            case "parallel":
                StatParallel.run(nGramSize, filePath);
                break;
            default:
                System.out.println("Unknown mode: " + mode + ". Supported modes: sequential, parallel");
                return;
        }

    }
}

