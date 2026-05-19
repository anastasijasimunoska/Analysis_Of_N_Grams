package com.project.nstat;

import mpi.MPI;

public class MainDistributed {

    public static void main(String[] args) throws Exception {
        MPI.Init(args);

        int rank = MPI.COMM_WORLD.Rank();

        if (args.length < 6) {
            if (rank == 0) {
                System.out.println("Usage: java -jar starter.jar " +
                        "-np <N> com.project.nstat.MainDistributed " +
                        "distributed <nGramSize> <filePath>");
            }
            MPI.Finalize();
            return;
        }

        String mode = args[3].toLowerCase();
        String nGramStr = args[4];
        String filePath = args[5];

        int nGramSize;
        try {
            nGramSize = Integer.parseInt(nGramStr);
        } catch (NumberFormatException e) {
            if (rank == 0) {
                System.out.println(
                        "Invalid n-gram size. Must be an integer.");
            }
            MPI.Finalize();
            return;
        }

        if (mode.equals("distributed")) {
            StatDistributed.run(nGramSize, filePath);
        } else {
            if (rank == 0) {
                System.out.println(
                        "Unsupported mode: " + mode);
            }
            MPI.Finalize();
            return;
        }

        MPI.Finalize();
    }
}
