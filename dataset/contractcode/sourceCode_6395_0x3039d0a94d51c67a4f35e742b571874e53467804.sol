/**
 *Submitted for verification at Etherscan.io on 2016-07-06
*/

contract Vote {
    event LogVote(address indexed addr);

    function() {
        LogVote(msg.sender);

        if (msg.value > 0) {
            msg.sender.send(msg.value);
        }
    }
}